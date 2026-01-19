# How Long Is a Piece of String? A Brief Empirical Analysis of Tokenizers 

**Authors**: Jonathan Roberts, Kai Han, Samuel Albanie  

**Link**: [PDF](https://arxiv.org/pdf/2601.11518)  

**Abstract**: Frontier LLMs are increasingly utilised across academia, society and industry. A commonly used unit for comparing models, their inputs and outputs, and estimating inference pricing is the token. In general, tokens are used as a stable currency, assumed to be broadly consistent across tokenizers and contexts, enabling direct comparisons. However, tokenization varies significantly across models and domains of text, making naive interpretation of token counts problematic. We quantify this variation by providing a comprehensive empirical analysis of tokenization, exploring the compression of sequences to tokens across different distributions of textual data. Our analysis challenges commonly held heuristics about token lengths, finding them to be overly simplistic. We hope the insights of our study add clarity and intuition toward tokenization in contemporary LLMs. 

---
# Do explanations generalize across large reasoning models? 

**Authors**: Koyena Pal, David Bau, Chandan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2601.11517)  

**Abstract**: Large reasoning models (LRMs) produce a textual chain of thought (CoT) in the process of solving a problem, which serves as a potentially powerful tool to understand the problem by surfacing a human-readable, natural-language explanation. However, it is unclear whether these explanations generalize, i.e. whether they capture general patterns about the underlying problem rather than patterns which are esoteric to the LRM. This is a crucial question in understanding or discovering new concepts, e.g. in AI for science. We study this generalization question by evaluating a specific notion of generalizability: whether explanations produced by one LRM induce the same behavior when given to other LRMs. We find that CoT explanations often exhibit this form of generalization (i.e. they increase consistency between LRMs) and that this increased generalization is correlated with human preference rankings and post-training with reinforcement learning. We further analyze the conditions under which explanations yield consistent answers and propose a straightforward, sentence-level ensembling strategy that improves consistency. Taken together, these results prescribe caution when using LRM explanations to yield new insights and outline a framework for characterizing LRM explanation generalization. 

---
# CTest-Metric: A Unified Framework to Assess Clinical Validity of Metrics for CT Report Generation 

**Authors**: Vanshali Sharma, Andrea Mia Bejar, Gorkem Durak, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2601.11488)  

**Abstract**: In the generative AI era, where even critical medical tasks are increasingly automated, radiology report generation (RRG) continues to rely on suboptimal metrics for quality assessment. Developing domain-specific metrics has therefore been an active area of research, yet it remains challenging due to the lack of a unified, well-defined framework to assess their robustness and applicability in clinical contexts. To address this, we present CTest-Metric, a first unified metric assessment framework with three modules determining the clinical feasibility of metrics for CT RRG. The modules test: (i) Writing Style Generalizability (WSG) via LLM-based rephrasing; (ii) Synthetic Error Injection (SEI) at graded severities; and (iii) Metrics-vs-Expert correlation (MvE) using clinician ratings on 175 "disagreement" cases. Eight widely used metrics (BLEU, ROUGE, METEOR, BERTScore-F1, F1-RadGraph, RaTEScore, GREEN Score, CRG) are studied across seven LLMs built on a CT-CLIP encoder. Using our novel framework, we found that lexical NLG metrics are highly sensitive to stylistic variations; GREEN Score aligns best with expert judgments (Spearman~0.70), while CRG shows negative correlation; and BERTScore-F1 is least sensitive to factual error injection. We will release the framework, code, and allowable portion of the anonymized evaluation data (rephrased/error-injected CT reports), to facilitate reproducible benchmarking and future metric development. 

---
# Predict the Retrieval! Test time adaptation for Retrieval Augmented Generation 

**Authors**: Xin Sun, Zhongqi Chen, Qiang Liu, Shu Wu, Bowen Song, Weiqiang Wang, Zilei Wang, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11443)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful approach for enhancing large language models' question-answering capabilities through the integration of external knowledge. However, when adapting RAG systems to specialized domains, challenges arise from distribution shifts, resulting in suboptimal generalization performance. In this work, we propose TTARAG, a test-time adaptation method that dynamically updates the language model's parameters during inference to improve RAG system performance in specialized domains. Our method introduces a simple yet effective approach where the model learns to predict retrieved content, enabling automatic parameter adjustment to the target domain. Through extensive experiments across six specialized domains, we demonstrate that TTARAG achieves substantial performance improvements over baseline RAG systems. Code available at this https URL. 

---
# Hierarchical Orthogonal Residual Spread for Precise Massive Editing in Large Language Models 

**Authors**: Xiaojie Gu, Guangxu Chen, Yuheng Yang, Jingxin Han, Andi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11441)  

**Abstract**: Large language models (LLMs) exhibit exceptional performance across various domains, yet they face critical safety concerns. Model editing has emerged as an effective approach to mitigate these issues. Existing model editing methods often focus on optimizing an information matrix that blends new and old knowledge. While effective, these approaches can be computationally expensive and may cause conflicts. In contrast, we shift our attention to Hierarchical Orthogonal Residual SprEad of the information matrix, which reduces noisy gradients and enables more stable edits from a different perspective. We demonstrate the effectiveness of our method HORSE through a clear theoretical comparison with several popular methods and extensive experiments conducted on two datasets across multiple LLMs. The results show that HORSE maintains precise massive editing across diverse scenarios. The code is available at this https URL 

---
# The unreasonable effectiveness of pattern matching 

**Authors**: Gary Lupyan, Blaise Agüera y Arcas  

**Link**: [PDF](https://arxiv.org/pdf/2601.11432)  

**Abstract**: We report on an astonishing ability of large language models (LLMs) to make sense of "Jabberwocky" language in which most or all content words have been randomly replaced by nonsense strings, e.g., translating "He dwushed a ghanc zawk" to "He dragged a spare chair". This result addresses ongoing controversies regarding how to best think of what LLMs are doing: are they a language mimic, a database, a blurry version of the Web? The ability of LLMs to recover meaning from structural patterns speaks to the unreasonable effectiveness of pattern-matching. Pattern-matching is not an alternative to "real" intelligence, but rather a key ingredient. 

---
# Relational Linearity is a Predictor of Hallucinations 

**Authors**: Yuetian Lu, Yihong Liu, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2601.11429)  

**Abstract**: Hallucination is a central failure mode in large language models (LLMs). We focus on hallucinations of answers to questions like: "Which instrument did Glenn Gould play?", but we ask these questions for synthetic entities that are unknown to the model. Surprisingly, we find that medium-size models like Gemma-7B-IT frequently hallucinate, i.e., they have difficulty recognizing that the hallucinated fact is not part of their knowledge. We hypothesize that an important factor in causing these hallucinations is the linearity of the relation: linear relations tend to be stored more abstractly, making it difficult for the LLM to assess its knowledge; the facts of nonlinear relations tend to be stored more directly, making knowledge assessment easier. To investigate this hypothesis, we create SyntHal, a dataset of 6000 synthetic entities for six relations. In our experiments with four models, we determine, for each relation, the hallucination rate on SyntHal and also measure its linearity, using $\Delta\cos$. We find a strong correlation ($r \in [.78,.82]$) between relational linearity and hallucination rate, providing evidence for our hypothesis that the underlying storage of triples of a relation is a factor in how well a model can self-assess its knowledge. This finding has implications for how to manage hallucination behavior and suggests new research directions for improving the representation of factual knowledge in LLMs. 

---
# Evaluating LLM Behavior in Hiring: Implicit Weights, Fairness Across Groups, and Alignment with Human Preferences 

**Authors**: Morgane Hoffmann, Emma Jouffroy, Warren Jouanneau, Marc Palyart, Charles Pebereau  

**Link**: [PDF](https://arxiv.org/pdf/2601.11379)  

**Abstract**: General-purpose Large Language Models (LLMs) show significant potential in recruitment applications, where decisions require reasoning over unstructured text, balancing multiple criteria, and inferring fit and competence from indirect productivity signals. Yet, it is still uncertain how LLMs assign importance to each attribute and whether such assignments are in line with economic principles, recruiter preferences or broader societal norms. We propose a framework to evaluate an LLM's decision logic in recruitment, by drawing on established economic methodologies for analyzing human hiring behavior. We build synthetic datasets from real freelancer profiles and project descriptions from a major European online freelance marketplace and apply a full factorial design to estimate how a LLM weighs different match-relevant criteria when evaluating freelancer-project fit. We identify which attributes the LLM prioritizes and analyze how these weights vary across project contexts and demographic subgroups. Finally, we explain how a comparable experimental setup could be implemented with human recruiters to assess alignment between model and human decisions. Our findings reveal that the LLM weighs core productivity signals, such as skills and experience, but interprets certain features beyond their explicit matching value. While showing minimal average discrimination against minority groups, intersectional effects reveal that productivity signals carry different weights between demographic groups. 

---
# Reward Modeling for Scientific Writing Evaluation 

**Authors**: Furkan Şahinuç, Subhabrata Dutta, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2601.11374)  

**Abstract**: Scientific writing is an expert-domain task that demands deep domain knowledge, task-specific requirements and reasoning capabilities that leverage the domain knowledge to satisfy the task specifications. While scientific text generation has been widely studied, its evaluation remains a challenging and open problem. It is critical to develop models that can be reliably deployed for evaluating diverse open-ended scientific writing tasks while adhering to their distinct requirements. However, existing LLM-based judges and reward models are primarily optimized for general-purpose benchmarks with fixed scoring rubrics and evaluation criteria. Consequently, they often fail to reason over sparse knowledge of scientific domains when interpreting task-dependent and multi-faceted criteria. Moreover, fine-tuning for each individual task is costly and impractical for low-resource settings. To bridge these gaps, we propose cost-efficient, open-source reward models tailored for scientific writing evaluation. We introduce a two-stage training framework that initially optimizes scientific evaluation preferences and then refines reasoning capabilities. Our multi-aspect evaluation design and joint training across diverse tasks enable fine-grained assessment and robustness to dynamic criteria and scoring rubrics. Experimental analysis shows that our training regime strongly improves LLM-based scientific writing evaluation. Our models generalize effectively across tasks and to previously unseen scientific writing evaluation settings, allowing a single trained evaluator to be reused without task-specific retraining. 

---
# How Much Would a Clinician Edit This Draft? Evaluating LLM Alignment for Patient Message Response Drafting 

**Authors**: Parker Seegmiller, Joseph Gatto, Sarah E. Greer, Ganza Belise Isingizwe, Rohan Ray, Timothy E. Burdick, Sarah Masud Preum  

**Link**: [PDF](https://arxiv.org/pdf/2601.11344)  

**Abstract**: Large language models (LLMs) show promise in drafting responses to patient portal messages, yet their integration into clinical workflows raises various concerns, including whether they would actually save clinicians time and effort in their portal workload. We investigate LLM alignment with individual clinicians through a comprehensive evaluation of the patient message response drafting task. We develop a novel taxonomy of thematic elements in clinician responses and propose a novel evaluation framework for assessing clinician editing load of LLM-drafted responses at both content and theme levels. We release an expert-annotated dataset and conduct large-scale evaluations of local and commercial LLMs using various adaptation techniques including thematic prompting, retrieval-augmented generation, supervised fine-tuning, and direct preference optimization. Our results reveal substantial epistemic uncertainty in aligning LLM drafts with clinician responses. While LLMs demonstrate capability in drafting certain thematic elements, they struggle with clinician-aligned generation in other themes, particularly question asking to elicit further information from patients. Theme-driven adaptation strategies yield improvements across most themes. Our findings underscore the necessity of adapting LLMs to individual clinician preferences to enable reliable and responsible use in patient-clinician communication workflows. 

---
# Neural Chain-of-Thought Search: Searching the Optimal Reasoning Path to Enhance Large Language Models 

**Authors**: Guoming Ling, Zhongzhan Huang, Yupei Lin, Junxin Li, Shanshan Zhong, Hefeng Wu, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.11340)  

**Abstract**: Chain-of-Thought reasoning has significantly enhanced the problem-solving capabilities of Large Language Models. Unfortunately, current models generate reasoning steps sequentially without foresight, often becoming trapped in suboptimal reasoning paths with redundant steps. In contrast, we introduce Neural Chain-of-Thought Search (NCoTS), a framework that reformulates reasoning as a dynamic search for the optimal thinking strategy. By quantitatively characterizing the solution space, we reveal the existence of sparse superior reasoning paths that are simultaneously more accurate and concise than standard outputs. Our method actively navigates towards these paths by evaluating candidate reasoning operators using a dual-factor heuristic that optimizes for both correctness and computational cost. Consequently, NCoTS achieves a Pareto improvement across diverse reasoning benchmarks, boosting accuracy by over 3.5% while reducing generation length by over 22%. Our code and data are available at this https URL. 

---
# Idea First, Code Later: Disentangling Problem Solving from Code Generation in Evaluating LLMs for Competitive Programming 

**Authors**: Sama Hadhoud, Alaa Elsetohy, Frederikus Hudi, Jan Christian Blaise Cruz, Steven Halim, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2601.11332)  

**Abstract**: Large Language Models (LLMs) increasingly succeed on competitive programming problems, yet existing evaluations conflate algorithmic reasoning with code-level implementation. We argue that competitive programming is fundamentally a problem-solving task and propose centering natural-language editorials in both solution generation and evaluation. Generating an editorial prior to code improves solve rates for some LLMs, with substantially larger gains when using expertly written gold editorials. However, even with gold editorials, models continue to struggle with implementation, while the gap between generated and gold editorials reveals a persistent problem-solving bottleneck in specifying correct and complete algorithms. Beyond pass/fail metrics, we diagnose reasoning errors by comparing model-generated editorials to gold standards using expert annotations and validate an LLM-as-a-judge protocol for scalable evaluation. We introduce a dataset of 83 ICPC-style problems with gold editorials and full test suites, and evaluate 19 LLMs, arguing that future benchmarks should explicitly separate problem solving from implementation. 

---
# F-Actor: Controllable Conversational Behaviour in Full-Duplex Models 

**Authors**: Maike Züfle, Ondrej Klejch, Nicholas Sanders, Jan Niehues, Alexandra Birch, Tsz Kin Lam  

**Link**: [PDF](https://arxiv.org/pdf/2601.11329)  

**Abstract**: Spoken conversational systems require more than accurate speech generation to have human-like conversations: to feel natural and engaging, they must produce conversational behaviour that adapts dynamically to the context. Current spoken conversational systems, however, rarely allow such customization, limiting their naturalness and usability. In this work, we present the first open, instruction-following full-duplex conversational speech model that can be trained efficiently under typical academic resource constraints. By keeping the audio encoder frozen and finetuning only the language model, our model requires just 2,000 hours of data, without relying on large-scale pretraining or multi-stage optimization. The model can follow explicit instructions to control speaker voice, conversation topic, conversational behaviour (e.g., backchanneling and interruptions), and dialogue initiation. We propose a single-stage training protocol and systematically analyze design choices. Both the model and training code will be released to enable reproducible research on controllable full-duplex speech systems. 

---
# Membership Inference on LLMs in the Wild 

**Authors**: Jiatong Yi, Yanyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.11314)  

**Abstract**: Membership Inference Attacks (MIAs) act as a crucial auditing tool for the opaque training data of Large Language Models (LLMs). However, existing techniques predominantly rely on inaccessible model internals (e.g., logits) or suffer from poor generalization across domains in strict black-box settings where only generated text is available. In this work, we propose SimMIA, a robust MIA framework tailored for this text-only regime by leveraging an advanced sampling strategy and scoring mechanism. Furthermore, we present WikiMIA-25, a new benchmark curated to evaluate MIA performance on modern proprietary LLMs. Experiments demonstrate that SimMIA achieves state-of-the-art results in the black-box setting, rivaling baselines that exploit internal model information. 

---
# One LLM to Train Them All: Multi-Task Learning Framework for Fact-Checking 

**Authors**: Malin Astrid Larsson, Harald Fosen Grunnaleite, Vinay Setty  

**Link**: [PDF](https://arxiv.org/pdf/2601.11293)  

**Abstract**: Large language models (LLMs) are reshaping automated fact-checking (AFC) by enabling unified, end-to-end verification pipelines rather than isolated components. While large proprietary models achieve strong performance, their closed weights, complexity, and high costs limit sustainability. Fine-tuning smaller open weight models for individual AFC tasks can help but requires multiple specialized models resulting in high costs. We propose \textbf{multi-task learning (MTL)} as a more efficient alternative that fine-tunes a single model to perform claim detection, evidence ranking, and stance detection jointly. Using small decoder-only LLMs (e.g., Qwen3-4b), we explore three MTL strategies: classification heads, causal language modeling heads, and instruction-tuning, and evaluate them across model sizes, task orders, and standard non-LLM baselines. While multitask models do not universally surpass single-task baselines, they yield substantial improvements, achieving up to \textbf{44\%}, \textbf{54\%}, and \textbf{31\%} relative gains for claim detection, evidence re-ranking, and stance detection, respectively, over zero-/few-shot settings. Finally, we also provide practical, empirically grounded guidelines to help practitioners apply MTL with LLMs for automated fact-checking. 

---
# Reasoning in Trees: Improving Retrieval-Augmented Generation for Multi-Hop Question Answering 

**Authors**: Yuling Shi, Maolin Sun, Zijun Liu, Mo Yang, Yixiong Fang, Tianran Sun, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11255)  

**Abstract**: Retrieval-Augmented Generation (RAG) has demonstrated significant effectiveness in enhancing large language models (LLMs) for complex multi-hop question answering (QA). For multi-hop QA tasks, current iterative approaches predominantly rely on LLMs to self-guide and plan multi-step exploration paths during retrieval, leading to substantial challenges in maintaining reasoning coherence across steps from inaccurate query decomposition and error propagation. To address these issues, we introduce Reasoning Tree Guided RAG (RT-RAG), a novel hierarchical framework for complex multi-hop QA. RT-RAG systematically decomposes multi-hop questions into explicit reasoning trees, minimizing inaccurate decomposition through structured entity analysis and consensus-based tree selection that clearly separates core queries, known entities, and unknown entities. Subsequently, a bottom-up traversal strategy employs iterative query rewriting and refinement to collect high-quality evidence, thereby mitigating error propagation. Comprehensive experiments show that RT-RAG substantially outperforms state-of-the-art methods by 7.0% F1 and 6.0% EM, demonstrating the effectiveness of RT-RAG in complex multi-hop QA. 

---
# How DDAIR you? Disambiguated Data Augmentation for Intent Recognition 

**Authors**: Galo Castillo-López, Alexis Lombard, Nasredine Semmar, Gaël de Chalendar  

**Link**: [PDF](https://arxiv.org/pdf/2601.11234)  

**Abstract**: Large Language Models (LLMs) are effective for data augmentation in classification tasks like intent detection. In some cases, they inadvertently produce examples that are ambiguous with regard to untargeted classes. We present DDAIR (Disambiguated Data Augmentation for Intent Recognition) to mitigate this problem. We use Sentence Transformers to detect ambiguous class-guided augmented examples generated by LLMs for intent recognition in low-resource scenarios. We identify synthetic examples that are semantically more similar to another intent than to their target one. We also provide an iterative re-generation method to mitigate such ambiguities. Our findings show that sentence embeddings effectively help to (re)generate less ambiguous examples, and suggest promising potential to improve classification performance in scenarios where intents are loosely or broadly defined. 

---
# FactCorrector: A Graph-Inspired Approach to Long-Form Factuality Correction of Large Language Models 

**Authors**: Javier Carnerero-Cano, Massimiliano Pronesti, Radu Marinescu, Tigran Tchrakian, James Barry, Jasmina Gajcin, Yufang Hou, Alessandra Pascale, Elizabeth Daly  

**Link**: [PDF](https://arxiv.org/pdf/2601.11232)  

**Abstract**: Large language models (LLMs) are widely used in knowledge-intensive applications but often generate factually incorrect responses. A promising approach to rectify these flaws is correcting LLMs using feedback. Therefore, in this paper, we introduce FactCorrector, a new post-hoc correction method that adapts across domains without retraining and leverages structured feedback about the factuality of the original response to generate a correction. To support rigorous evaluations of factuality correction methods, we also develop the VELI5 benchmark, a novel dataset containing systematically injected factual errors and ground-truth corrections. Experiments on VELI5 and several popular long-form factuality datasets show that the FactCorrector approach significantly improves factual precision while preserving relevance, outperforming strong baselines. We release our code at this https URL. 

---
# Language of Thought Shapes Output Diversity in Large Language Models 

**Authors**: Shaoyang Xu, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11227)  

**Abstract**: Output diversity is crucial for Large Language Models as it underpins pluralism and creativity. In this work, we reveal that controlling the language used during model thinking-the language of thought-provides a novel and structural source of output diversity. Our preliminary study shows that different thinking languages occupy distinct regions in a model's thinking space. Based on this observation, we study two repeated sampling strategies under multilingual thinking-Single-Language Sampling and Mixed-Language Sampling-and conduct diversity evaluation on outputs that are controlled to be in English, regardless of the thinking language used. Across extensive experiments, we demonstrate that switching the thinking language from English to non-English languages consistently increases output diversity, with a clear and consistent positive correlation such that languages farther from English in the thinking space yield larger gains. We further show that aggregating samples across multiple thinking languages yields additional improvements through compositional effects, and that scaling sampling with linguistic heterogeneity expands the model's diversity ceiling. Finally, we show that these findings translate into practical benefits in pluralistic alignment scenarios, leading to broader coverage of cultural knowledge and value orientations in LLM outputs. Our code is publicly available at this https URL. 

---
# MultiCaption: Detecting disinformation using multilingual visual claims 

**Authors**: Rafael Martins Frade, Rrubaa Panchendrarajan, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2601.11220)  

**Abstract**: Online disinformation poses an escalating threat to society, driven increasingly by the rapid spread of misleading content across both multimedia and multilingual platforms. While automated fact-checking methods have advanced in recent years, their effectiveness remains constrained by the scarcity of datasets that reflect these real-world complexities. To address this gap, we first present MultiCaption, a new dataset specifically designed for detecting contradictions in visual claims. Pairs of claims referring to the same image or video were labeled through multiple strategies to determine whether they contradict each other. The resulting dataset comprises 11,088 visual claims in 64 languages, offering a unique resource for building and evaluating misinformation-detection systems in truly multimodal and multilingual environments. We then provide comprehensive experiments using transformer-based architectures, natural language inference models, and large language models, establishing strong baselines for future research. The results show that MultiCaption is more challenging than standard NLI tasks, requiring task-specific finetuning for strong performance. Moreover, the gains from multilingual training and testing highlight the dataset's potential for building effective multilingual fact-checking pipelines without relying on machine translation. 

---
# T$^\star$: Progressive Block Scaling for MDM Through Trajectory Aware RL 

**Authors**: Hanchen Xia, Baoyou Chen, Yutang Ge, Guojiang Zhao, Siyu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11214)  

**Abstract**: We present T$^\star$, a simple \textsc{TraceRL}-based training curriculum for progressive block-size scaling in masked diffusion language models (MDMs). Starting from an AR-initialized small-block MDM, T$^\star$~transitions smoothly to larger blocks, enabling higher-parallelism decoding with minimal performance degradation on math reasoning benchmarks. Moreover, further analysis suggests that T$^\star$~can converge to an alternative decoding schedule $\hat{\rm S}$ that achieves comparable performance. 

---
# DOREMI: Optimizing Long Tail Predictions in Document-Level Relation Extraction 

**Authors**: Laura Menotti, Stefano Marchesin, Gianmaria Silvello  

**Link**: [PDF](https://arxiv.org/pdf/2601.11190)  

**Abstract**: Document-Level Relation Extraction (DocRE) presents significant challenges due to its reliance on cross-sentence context and the long-tail distribution of relation types, where many relations have scarce training examples. In this work, we introduce DOcument-level Relation Extraction optiMizing the long taIl (DOREMI), an iterative framework that enhances underrepresented relations through minimal yet targeted manual annotations. Unlike previous approaches that rely on large-scale noisy data or heuristic denoising, DOREMI actively selects the most informative examples to improve training efficiency and robustness. DOREMI can be applied to any existing DocRE model and is effective at mitigating long-tail biases, offering a scalable solution to improve generalization on rare relations. 

---
# The Growing Gains and Pains of Iterative Web Corpora Crawling: Insights from South Slavic CLASSLA-web 2.0 Corpora 

**Authors**: Taja Kuzman Pungeršek, Peter Rupnik, Vít Suchomel, Nikola Ljubešić  

**Link**: [PDF](https://arxiv.org/pdf/2601.11170)  

**Abstract**: Crawling national top-level domains has proven to be highly effective for collecting texts in less-resourced languages. This approach has been recently used for South Slavic languages and resulted in the largest general corpora for this language group: the CLASSLA-web 1.0 corpora. Building on this success, we established a continuous crawling infrastructure for iterative national top-level domain crawling across South Slavic and related webs. We present the first outcome of this crawling infrastructure - the CLASSLA-web 2.0 corpus collection, with substantially larger web corpora containing 17.0 billion words in 38.1 million texts in seven languages: Bosnian, Bulgarian, Croatian, Macedonian, Montenegrin, Serbian, and Slovenian. In addition to genre categories, the new version is also automatically annotated with topic labels. Comparing CLASSLA-web 2.0 with its predecessor reveals that only one-fifth of the texts overlap, showing that re-crawling after just two years yields largely new content. However, while the new web crawls bring growing gains, we also notice growing pains - a manual inspection of top domains reveals a visible degradation of web content, as machine-generated sites now contribute a significant portion of texts. 

---
# Integrity Shield A System for Ethical AI Use & Authorship Transparency in Assessments 

**Authors**: Ashish Raj Shekhar, Shiven Agarwal, Priyanuj Bordoloi, Yash Shah, Tejas Anvekar, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2601.11093)  

**Abstract**: Large Language Models (LLMs) can now solve entire exams directly from uploaded PDF assessments, raising urgent concerns about academic integrity and the reliability of grades and credentials. Existing watermarking techniques either operate at the token level or assume control over the model's decoding process, making them ineffective when students query proprietary black-box systems with instructor-provided documents. We present Integrity Shield, a document-layer watermarking system that embeds schema-aware, item-level watermarks into assessment PDFs while keeping their human-visible appearance unchanged. These watermarks consistently prevent MLLMs from answering shielded exam PDFs and encode stable, item-level signatures that can be reliably recovered from model or student responses. Across 30 exams spanning STEM, humanities, and medical reasoning, Integrity Shield achieves exceptionally high prevention (91-94% exam-level blocking) and strong detection reliability (89-93% signature retrieval) across four commercial MLLMs. Our demo showcases an interactive interface where instructors upload an exam, preview watermark behavior, and inspect pre/post AI performance & authorship evidence. 

---
# Efficient Multilingual Name Type Classification Using Convolutional Networks 

**Authors**: Davor Lauc  

**Link**: [PDF](https://arxiv.org/pdf/2601.11090)  

**Abstract**: We present a convolutional neural network approach for classifying proper names by language and entity type. Our model, Onomas-CNN X, combines parallel convolution branches with depthwise-separable operations and hierarchical classification to process names efficiently on CPU hardware. We evaluate the architecture on a large multilingual dataset covering 104 languages and four entity types (person, organization, location, other). Onomas-CNN X achieves 92.1% accuracy while processing 2,813 names per second on a single CPU core - 46 times faster than fine-tuned XLM-RoBERTa with comparable accuracy. The model reduces energy consumption by a factor of 46 compared to transformer baselines. Our experiments demonstrate that specialized CNN architectures remain competitive with large pre-trained models for focused NLP tasks when sufficient training data exists. 

---
# CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs 

**Authors**: Yuanxiang Liu, Songze Li, Xiaoke Guo, Zhaoyan Gong, Qifei Zhang, Huajun Chen, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11047)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities but often grapple with reliability challenges like hallucinations. While Knowledge Graphs (KGs) offer explicit grounding, existing paradigms of KG-augmented LLMs typically exhibit cognitive rigidity--applying homogeneous search strategies that render them vulnerable to instability under neighborhood noise and structural misalignment leading to reasoning stagnation. To address these challenges, we propose CoG, a training-free framework inspired by Dual-Process Theory that mimics the interplay between intuition and deliberation. First, functioning as the fast, intuitive process, the Relational Blueprint Guidance module leverages relational blueprints as interpretable soft structural constraints to rapidly stabilize the search direction against noise. Second, functioning as the prudent, analytical process, the Failure-Aware Refinement module intervenes upon encountering reasoning impasses. It triggers evidence-conditioned reflection and executes controlled backtracking to overcome reasoning stagnation. Experimental results on three benchmarks demonstrate that CoG significantly outperforms state-of-the-art approaches in both accuracy and efficiency. 

---
# Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse 

**Authors**: Chi Zhang, Mengqi Zhang, Xiaotian Ye, Runxi Cheng, Zisheng Zhou, Ying Zhou, Pengjie Ren, Zhumin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11042)  

**Abstract**: Sequential knowledge editing in large language models often causes catastrophic collapse of the model's general abilities, especially for parameter-modifying methods. Existing approaches mitigate this issue through heuristic constraints on parameter updates, yet the mechanisms underlying such degradation remain insufficiently understood. In this work, we present a spectral analysis of sequential knowledge editing and show that a model's general abilities are closely associated with dominant singular directions of pretrained weight matrices. These directions are highly sensitive to perturbations and are progressively disrupted by repeated edits, closely tracking the collapse in both editing efficacy and general performance. Building on this insight, we propose REVIVE, a plug-and-play framework that stabilizes sequential editing by explicitly preserving the dominant singular subspace. REVIVE represents parameter updates in the spectral basis of the original weights and filters components that would interfere with the protected region. Extensive experiments across multiple models and benchmarks show that REVIVE consistently improves editing efficacy while substantially preserving general abilities under long-horizon sequential editing, including extreme settings with up to 20,000 edits. 

---
# Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data 

**Authors**: Xuanming Zhang, Shwan Ashrafi, Aziza Mirsaidova, Amir Rezaeian, Miguel Ballesteros, Lydia B. Chilton, Zhou Yu, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2601.11038)  

**Abstract**: We study the reasoning behavior of large language models (LLMs) under limited computation budgets. In such settings, producing useful partial solutions quickly is often more practical than exhaustive reasoning, which incurs high inference costs. Many real-world tasks, such as trip planning, require models to deliver the best possible output within a fixed reasoning budget. We introduce an anytime reasoning framework and the Anytime Index, a metric that quantifies how effectively solution quality improves as reasoning tokens increase. To further enhance efficiency, we propose an inference-time self-improvement method using LLM-synthesized preference data, where models learn from their own reasoning comparisons to produce better intermediate solutions. Experiments on NaturalPlan (Trip), AIME, and GPQA datasets show consistent gains across Grok-3, GPT-oss, GPT-4.1/4o, and LLaMA models, improving both reasoning quality and efficiency under budget constraints. 

---
# From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models 

**Authors**: Youmi Ma, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2601.11020)  

**Abstract**: Advances in mechanistic interpretability have identified special attention heads, known as retrieval heads, that are responsible for retrieving information from the context. However, the role of these retrieval heads in improving model performance remains unexplored. This work investigates whether retrieval heads can be leveraged to enhance the long-context capabilities of LLMs. Specifically, we propose RetMask, a method that generates training signals by contrasting normal model outputs with those from an ablated variant in which the retrieval heads are masked. This mechanism-based approach achieves substantial improvements: +2.28 points on HELMET at 128K for Llama-3.1, with +70% gains on generation with citation and +32% on passage re-ranking, while preserving performance on general tasks. Experiments across three model families reveal that the effectiveness depends on retrieval head organization: models with concentrated patterns of retrieval heads respond strongly, while those with distributed patterns show limited gains. This mechanistic relationship validates the function of retrieval heads and demonstrates that mechanistic insights can be transformed into performance enhancements. 

---
# Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs 

**Authors**: Xinwei Wu, Heng Liu, Xiaohu Zhao, Yuqi Ren, Linlong Xu, Longyue Wang, Deyi Xiong, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11019)  

**Abstract**: Large Language Models (LLMs) frequently exhibit strong translation abilities, even without task-specific fine-tuning. However, the internal mechanisms governing this innate capability remain largely opaque. To demystify this process, we leverage Sparse Autoencoders (SAEs) and introduce a novel framework for identifying task-specific features. Our method first recalls features that are frequently co-activated on translation inputs and then filters them for functional coherence using a PCA-based consistency metric. This framework successfully isolates a small set of **translation initiation** features. Causal interventions demonstrate that amplifying these features steers the model towards correct translation, while ablating them induces hallucinations and off-task outputs, confirming they represent a core component of the model's innate translation competency. Moving from analysis to application, we leverage this mechanistic insight to propose a new data selection strategy for efficient fine-tuning. Specifically, we prioritize training on **mechanistically hard** samples-those that fail to naturally activate the translation initiation features. Experiments show this approach significantly improves data efficiency and suppresses hallucinations. Furthermore, we find these mechanisms are transferable to larger models of the same family. Our work not only decodes a core component of the translation mechanism in LLMs but also provides a blueprint for using internal model mechanism to create more robust and efficient models. The codes are available at this https URL. 

---
# NAACL: Noise-AwAre Verbal Confidence Calibration for LLMs in RAG Systems 

**Authors**: Jiayu Liu, Rui Wang, Qing Zong, Qingcheng Zeng, Tianshi Zheng, Haochen Shi, Dadi Guo, Baixuan Xu, Chunyang Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.11004)  

**Abstract**: Accurately assessing model confidence is essential for deploying large language models (LLMs) in mission-critical factual domains. While retrieval-augmented generation (RAG) is widely adopted to improve grounding, confidence calibration in RAG settings remains poorly understood. We conduct a systematic study across four benchmarks, revealing that LLMs exhibit poor calibration performance due to noisy retrieved contexts. Specifically, contradictory or irrelevant evidence tends to inflate the model's false certainty, leading to severe overconfidence. To address this, we propose NAACL Rules (Noise-AwAre Confidence CaLibration Rules) to provide a principled foundation for resolving overconfidence under noise. We further design NAACL, a noise-aware calibration framework that synthesizes supervision from about 2K HotpotQA examples guided by these rules. By performing supervised fine-tuning (SFT) with this data, NAACL equips models with intrinsic noise awareness without relying on stronger teacher models. Empirical results show that NAACL yields substantial gains, improving ECE scores by 10.9% in-domain and 8.0% out-of-domain. By bridging the gap between retrieval noise and verbal calibration, NAACL paves the way for both accurate and epistemically reliable LLMs. 

---
# Redefining Machine Simultaneous Interpretation: From Incremental Translation to Human-Like Strategies 

**Authors**: Qianen Zhang, Zeyu Yang, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2601.11002)  

**Abstract**: Simultaneous Machine Translation (SiMT) requires high-quality translations under strict real-time constraints, which traditional policies with only READ/WRITE actions cannot fully address. We extend the action space of SiMT with four adaptive actions: Sentence_Cut, Drop, Partial_Summarization and Pronominalization, which enable real-time restructuring, omission, and simplification while preserving semantic fidelity. We adapt these actions in a large language model (LLM) framework and construct training references through action-aware prompting. To evaluate both quality and word-level monotonicity, we further develop a latency-aware TTS pipeline that maps textual outputs to speech with realistic timing. Experiments on the ACL60/60 English-Chinese, English-German and English-Japanese benchmarks show that our framework consistently improves semantic metrics and achieves lower delay compared to reference translations and salami-based baselines. Notably, combining Drop and Sentence_Cut leads to consistent improvements in the balance between fluency and latency. These results demonstrate that enriching the action space of LLM-based SiMT provides a promising direction for bridging the gap between human and machine interpretation. 

---
# When Personalization Misleads: Understanding and Mitigating Hallucinations in Personalized LLMs 

**Authors**: Zhongxiang Sun, Yi Zhan, Chenglei Shen, Weijie Yu, Xiao Zhang, Ming He, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11000)  

**Abstract**: Personalized large language models (LLMs) adapt model behavior to individual users to enhance user satisfaction, yet personalization can inadvertently distort factual reasoning. We show that when personalized LLMs face factual queries, there exists a phenomenon where the model generates answers aligned with a user's prior history rather than the objective truth, resulting in personalization-induced hallucinations that degrade factual reliability and may propagate incorrect beliefs, due to representational entanglement between personalization and factual representations. To address this issue, we propose Factuality-Preserving Personalized Steering (FPPS), a lightweight inference-time approach that mitigates personalization-induced factual distortions while preserving personalized behavior. We further introduce PFQABench, the first benchmark designed to jointly evaluate factual and personalized question answering under personalization. Experiments across multiple LLM backbones and personalization methods show that FPPS substantially improves factual accuracy while maintaining personalized performance. 

---
# ZPD Detector: Data Selection via Capability-Difficulty Alignment for Large Language Models 

**Authors**: Bo Yang, Yunkui Chen, Lanfei Feng, Yu Zhang, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.10986)  

**Abstract**: As the cost of training large language models continues to increase and high-quality training data become increasingly scarce, selecting high-value samples or synthesizing effective training data under limited data budgets has emerged as a critical research problem. Most existing data selection methods rely on static criteria, such as difficulty, uncertainty, or heuristics, and fail to model the evolving relationship between the model and the data. Inspired by the educational theory of the Zone of Proximal Development (ZPD), we propose ZPD Detector, a data selection framework that adopts a bidirectional perspective between models and data by explicitly modeling the alignment between sample difficulty and the model's current capability. ZPD Detector integrates difficulty calibration, model capability estimation based on Item Response Theory (IRT), and a capability-difficulty matching score to dynamically identify the most informative samples at each learning stage, improving data utilization efficiency; moreover, this dynamic matching strategy provides new insights into training strategy design. All code and data will be released after our work be accepted to support reproducible researc 

---
# Steering Language Models Before They Speak: Logit-Level Interventions 

**Authors**: Hyeseon An, Shinwoo Park, Hyundong Jin, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.10960)  

**Abstract**: Steering LLMs is essential for specialized applications such as style-sensitive text rewriting, user-adaptive communication, and toxicity mitigation. Current steering methods, such as prompting-based and activation-based approaches, are widely used to guide model behavior. However, activation-based techniques require deep access to internal layers, while prompting-based steering often fails to provide consistent or fine-grained control. In order to address these limitations, we propose a training-free inference-time logit intervention for controllable generation. Our approach utilizes a statistical token score table derived from z-normalized log-odds of labeled corpora to shift the decoding distribution. Empirical evaluations across three diverse datasets focusing on writing complexity, formality, and toxicity demonstrate that our method effectively steers output characteristics, confirming its broad applicability and task-agnostic nature. Our results show that statistically grounded logit steering can achieve large, consistent, and multi-task control gains: up to +47%p accuracy and 50x f1 improvement. 

---
# Multi-Stage Patient Role-Playing Framework for Realistic Clinical Interactions 

**Authors**: Shijie Jiang, Zefan Zhang, Kehua Zhu, Tian Bai, Ruihong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.10951)  

**Abstract**: The simulation of realistic clinical interactions plays a pivotal role in advancing clinical Large Language Models (LLMs) and supporting medical diagnostic education. Existing approaches and benchmarks rely on generic or LLM-generated dialogue data, which limits the authenticity and diversity of doctor-patient interactions. In this work, we propose the first Chinese patient simulation dataset (Ch-PatientSim), constructed from realistic clinical interaction scenarios to comprehensively evaluate the performance of models in emulating patient behavior. Patients are simulated based on a five-dimensional persona structure. To address issues of the persona class imbalance, a portion of the dataset is augmented using few-shot generation, followed by manual verification. We evaluate various state-of-the-art LLMs and find that most produce overly formal responses that lack individual personality. To address this limitation, we propose a training-free Multi-Stage Patient Role-Playing (MSPRP) framework, which decomposes interactions into three stages to ensure both personalization and realism in model responses. Experimental results demonstrate that our approach significantly improves model performance across multiple dimensions of patient simulation. 

---
# Selecting Language Models for Social Science: Start Small, Start Open, and Validate 

**Authors**: Dustin S. Stoltz, Marshall A. Taylor, Sanuj Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.10926)  

**Abstract**: Currently, there are thousands of large pretrained language models (LLMs) available to social scientists. How do we select among them? Using validity, reliability, reproducibility, and replicability as guides, we explore the significance of: (1) model openness, (2) model footprint, (3) training data, and (4) model architectures and fine-tuning. While ex-ante tests of validity (i.e., benchmarks) are often privileged in these discussions, we argue that social scientists cannot altogether avoid validating computational measures (ex-post). Replicability, in particular, is a more pressing guide for selecting language models. Being able to reliably replicate a particular finding that entails the use of a language model necessitates reliably reproducing a task. To this end, we propose starting with smaller, open models, and constructing delimited benchmarks to demonstrate the validity of the entire computational pipeline. 

---
# Massively Multilingual Joint Segmentation and Glossing 

**Authors**: Michael Ginn, Lindia Tjuatja, Enora Rice, Ali Marashian, Maria Valentini, Jasmine Xu, Graham Neubig, Alexis Palmer  

**Link**: [PDF](https://arxiv.org/pdf/2601.10925)  

**Abstract**: Automated interlinear gloss prediction with neural networks is a promising approach to accelerate language documentation efforts. However, while state-of-the-art models like GlossLM achieve high scores on glossing benchmarks, user studies with linguists have found critical barriers to the usefulness of such models in real-world scenarios. In particular, existing models typically generate morpheme-level glosses but assign them to whole words without predicting the actual morpheme boundaries, making the predictions less interpretable and thus untrustworthy to human annotators.
We conduct the first study on neural models that jointly predict interlinear glosses and the corresponding morphological segmentation from raw text. We run experiments to determine the optimal way to train models that balance segmentation and glossing accuracy, as well as the alignment between the two tasks. We extend the training corpus of GlossLM and pretrain PolyGloss, a family of seq2seq multilingual models for joint segmentation and glossing that outperforms GlossLM on glossing and beats various open-source LLMs on segmentation, glossing, and alignment. In addition, we demonstrate that PolyGloss can be quickly adapted to a new dataset via low-rank adaptation. 

---
# Neural Induction of Finite-State Transducers 

**Authors**: Michael Ginn, Alexis Palmer, Mans Hulden  

**Link**: [PDF](https://arxiv.org/pdf/2601.10918)  

**Abstract**: Finite-State Transducers (FSTs) are effective models for string-to-string rewriting tasks, often providing the efficiency necessary for high-performance applications, but constructing transducers by hand is difficult. In this work, we propose a novel method for automatically constructing unweighted FSTs following the hidden state geometry learned by a recurrent neural network. We evaluate our methods on real-world datasets for morphological inflection, grapheme-to-phoneme prediction, and historical normalization, showing that the constructed FSTs are highly accurate and robust for many datasets, substantially outperforming classical transducer learning algorithms by up to 87% accuracy on held-out test sets. 

---
# DialDefer: A Framework for Detecting and Mitigating LLM Dialogic Deference 

**Authors**: Parisa Rabbani, Priyam Sahoo, Ruben Mathew, Aishee Mondal, Harshita Ketharaman, Nimet Beyza Bozdag, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2601.10896)  

**Abstract**: LLMs are increasingly used as third-party judges, yet their reliability when evaluating speakers in dialogue remains poorly understood. We show that LLMs judge identical claims differently depending on framing: the same content elicits different verdicts when presented as a statement to verify ("Is this statement correct?") versus attributed to a speaker ("Is this speaker correct?"). We call this dialogic deference and introduce DialDefer, a framework for detecting and mitigating these framing-induced judgment shifts. Our Dialogic Deference Score (DDS) captures directional shifts that aggregate accuracy obscures. Across nine domains, 3k+ instances, and four models, conversational framing induces large shifts (|DDS| up to 87pp, p < .0001) while accuracy remains stable (<2pp), with effects amplifying 2-4x on naturalistic Reddit conversations. Models can shift toward agreement (deference) or disagreement (skepticism) depending on domain -- the same model ranges from DDS = -53 on graduate-level science to +58 on social judgment. Ablations reveal that human-vs-LLM attribution drives the largest shifts (17.7pp swing), suggesting models treat disagreement with humans as more costly than with AI. Mitigation attempts reduce deference but can over-correct into skepticism, framing this as a calibration problem beyond accuracy optimization. 

---
# EncodeRec: An Embedding Backbone for Recommendation Systems 

**Authors**: Guy Hadad, Neomi Rabaev, Bracha Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2601.10837)  

**Abstract**: Recent recommender systems increasingly leverage embeddings from large pre-trained language models (PLMs). However, such embeddings exhibit two key limitations: (1) PLMs are not explicitly optimized to produce structured and discriminative embedding spaces, and (2) their representations remain overly generic, often failing to capture the domain-specific semantics crucial for recommendation tasks. We present EncodeRec, an approach designed to align textual representations with recommendation objectives while learning compact, informative embeddings directly from item descriptions. EncodeRec keeps the language model parameters frozen during recommender system training, making it computationally efficient without sacrificing semantic fidelity. Experiments across core recommendation benchmarks demonstrate its effectiveness both as a backbone for sequential recommendation models and for semantic ID tokenization, showing substantial gains over PLM-based and embedding model baselines. These results underscore the pivotal role of embedding adaptation in bridging the gap between general-purpose language models and practical recommender systems. 

---
# Reasoning Models Generate Societies of Thought 

**Authors**: Junsol Kim, Shiyang Lai, Nino Scherrer, Blaise Agüera y Arcas, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2601.10825)  

**Abstract**: Large language models have achieved remarkable capabilities across domains, yet mechanisms underlying sophisticated reasoning remain elusive. Recent reasoning models outperform comparable instruction-tuned models on complex cognitive tasks, attributed to extended computation through longer chains of thought. Here we show that enhanced reasoning emerges not from extended computation alone, but from simulating multi-agent-like interactions -- a society of thought -- which enables diversification and debate among internal cognitive perspectives characterized by distinct personality traits and domain expertise. Through quantitative analysis and mechanistic interpretability methods applied to reasoning traces, we find that reasoning models like DeepSeek-R1 and QwQ-32B exhibit much greater perspective diversity than instruction-tuned models, activating broader conflict between heterogeneous personality- and expertise-related features during reasoning. This multi-agent structure manifests in conversational behaviors, including question-answering, perspective shifts, and the reconciliation of conflicting views, and in socio-emotional roles that characterize sharp back-and-forth conversations, together accounting for the accuracy advantage in reasoning tasks. Controlled reinforcement learning experiments reveal that base models increase conversational behaviors when rewarded solely for reasoning accuracy, and fine-tuning models with conversational scaffolding accelerates reasoning improvement over base models. These findings indicate that the social organization of thought enables effective exploration of solution spaces. We suggest that reasoning models establish a computational parallel to collective intelligence in human groups, where diversity enables superior problem-solving when systematically structured, which suggests new opportunities for agent organization to harness the wisdom of crowds. 

---
# A Concise Agent is Less Expert: Revealing Side Effects of Using Style Features on Conversational Agents 

**Authors**: Young-Min Cho, Yuan Yuan, Sharath Chandra Guntuku, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2601.10809)  

**Abstract**: Style features such as friendly, helpful, or concise are widely used in prompts to steer the behavior of Large Language Model (LLM) conversational agents, yet their unintended side effects remain poorly understood. In this work, we present the first systematic study of cross-feature stylistic side effects. We conduct a comprehensive survey of 127 conversational agent papers from ACL Anthology and identify 12 frequently used style features. Using controlled, synthetic dialogues across task-oriented and open domain settings, we quantify how prompting for one style feature causally affects others via a pairwise LLM as a Judge evaluation framework. Our results reveal consistent and structured side effects, such as prompting for conciseness significantly reduces perceived expertise. They demonstrate that style features are deeply entangled rather than orthogonal. To support future research, we introduce CASSE (Conversational Agent Stylistic Side Effects), a dataset capturing these complex interactions. We further evaluate prompt based and activation steering based mitigation strategies and find that while they can partially restore suppressed traits, they often degrade the primary intended style. These findings challenge the assumption of faithful style control in LLMs and highlight the need for multi-objective and more principled approaches to safe, targeted stylistic steering in conversational agents. 

---
# BYOL: Bring Your Own Language Into LLMs 

**Authors**: Syed Waqas Zamir, Wassim Hamidouche, Boulbaba Ben Amor, Luana Marotti, Inbal Becker-Reshef, Juan Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2601.10804)  

**Abstract**: Large Language Models (LLMs) exhibit strong multilingual capabilities, yet remain fundamentally constrained by the severe imbalance in global language resources. While over 7,000 languages are spoken worldwide, only a small subset (fewer than 100) has sufficient digital presence to meaningfully influence modern LLM training. This disparity leads to systematic underperformance, cultural misalignment, and limited accessibility for speakers of low-resource and extreme-low-resource languages. To address this gap, we introduce Bring Your Own Language (BYOL), a unified framework for scalable, language-aware LLM development tailored to each language's digital footprint. BYOL begins with a language resource classification that maps languages into four tiers (Extreme-Low, Low, Mid, High) using curated web-scale corpora, and uses this classification to select the appropriate integration pathway. For low-resource languages, we propose a full-stack data refinement and expansion pipeline that combines corpus cleaning, synthetic text generation, continual pretraining, and supervised finetuning. Applied to Chichewa and Maori, this pipeline yields language-specific LLMs that achieve approximately 12 percent average improvement over strong multilingual baselines across 12 benchmarks, while preserving English and multilingual capabilities via weight-space model merging. For extreme-low-resource languages, we introduce a translation-mediated inclusion pathway, and show on Inuktitut that a tailored machine translation system improves over a commercial baseline by 4 BLEU, enabling high-accuracy LLM access when direct language modeling is infeasible. Finally, we release human-translated versions of the Global MMLU-Lite benchmark in Chichewa, Maori, and Inuktitut, and make our codebase and models publicly available at this https URL . 

---
# LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning 

**Authors**: Tommaso Felice Banfi, Sashenka Gamage  

**Link**: [PDF](https://arxiv.org/pdf/2601.10775)  

**Abstract**: We propose a novel LLM-based framework for reasoning in discrete, game-theoretic tasks, illustrated with \emph{Tic-Tac-Toe}. The method integrates in-context learning with entropy-guided chain-of-thought (CoT) reasoning and adaptive context retrieval. The model dynamically adjusts both the number of retrieved examples and reasoning paths according to token-level uncertainty: concise reasoning with minimal context is used when uncertainty is low, whereas higher uncertainty triggers expanded multi-path CoT exploration. Experimental evaluation against a sub-optimal algorithmic opponent shows that entropy-aware adaptive reasoning substantially improves decision quality, increasing the average game outcome from \(-11.6\%\) with the baseline LLM to \(+9.5\%\) with entropy-guided adaptive reasoning over 100 games (win = +1, tie = 0, loss = -1), while maintaining a relatively low number of LLM queries per game. Statistical validation confirms that the improvement is significant, and correlation analysis reveals a negative association between token-level entropy and move optimality. These findings demonstrate that uncertainty-guided adaptive reasoning effectively enhances LLM performance in sequential decision-making environments. 

---
# Building Production-Ready Probes For Gemini 

**Authors**: János Kramár, Joshua Engels, Zheng Wang, Bilal Chughtai, Rohin Shah, Neel Nanda, Arthur Conmy  

**Link**: [PDF](https://arxiv.org/pdf/2601.11516)  

**Abstract**: Frontier language model capabilities are improving rapidly. We thus need stronger mitigations against bad actors misusing increasingly powerful systems. Prior work has shown that activation probes may be a promising misuse mitigation technique, but we identify a key remaining challenge: probes fail to generalize under important production distribution shifts. In particular, we find that the shift from short-context to long-context inputs is difficult for existing probe architectures. We propose several new probe architecture that handle this long-context distribution shift.
We evaluate these probes in the cyber-offensive domain, testing their robustness against various production-relevant shifts, including multi-turn conversations, static jailbreaks, and adaptive red teaming. Our results demonstrate that while multimax addresses context length, a combination of architecture choice and training on diverse distributions is required for broad generalization. Additionally, we show that pairing probes with prompted classifiers achieves optimal accuracy at a low cost due to the computational efficiency of probes.
These findings have informed the successful deployment of misuse mitigation probes in user-facing instances of Gemini, Google's frontier language model. Finally, we find early positive results using AlphaEvolve to automate improvements in both probe architecture search and adaptive red teaming, showing that automating some AI safety research is already possible. 

---
# The Poisoned Apple Effect: Strategic Manipulation of Mediated Markets via Technology Expansion of AI Agents 

**Authors**: Eilam Shapira, Roi Reichart, Moshe Tennenholtz  

**Link**: [PDF](https://arxiv.org/pdf/2601.11496)  

**Abstract**: The integration of AI agents into economic markets fundamentally alters the landscape of strategic interaction. We investigate the economic implications of expanding the set of available technologies in three canonical game-theoretic settings: bargaining (resource division), negotiation (asymmetric information trade), and persuasion (strategic information transmission). We find that simply increasing the choice of AI delegates can drastically shift equilibrium payoffs and regulatory outcomes, often creating incentives for regulators to proactively develop and release technologies. Conversely, we identify a strategic phenomenon termed the "Poisoned Apple" effect: an agent may release a new technology, which neither they nor their opponent ultimately uses, solely to manipulate the regulator's choice of market design in their favor. This strategic release improves the releaser's welfare at the expense of their opponent and the regulator's fairness objectives. Our findings demonstrate that static regulatory frameworks are vulnerable to manipulation via technology expansion, necessitating dynamic market designs that adapt to the evolving landscape of AI capabilities. 

---
# MHA2MLA-VLM: Enabling DeepSeek's Economical Multi-Head Latent Attention across Vision-Language Models 

**Authors**: Xiaoran Fan, Zhichao Sun, Tao Ji, Lixing Shen, Tao Gui  

**Link**: [PDF](https://arxiv.org/pdf/2601.11464)  

**Abstract**: As vision-language models (VLMs) tackle increasingly complex and multimodal tasks, the rapid growth of Key-Value (KV) cache imposes significant memory and computational bottlenecks during inference. While Multi-Head Latent Attention (MLA) offers an effective means to compress the KV cache and accelerate inference, adapting existing VLMs to the MLA architecture without costly pretraining remains largely unexplored. In this work, we present MHA2MLA-VLM, a parameter-efficient and multimodal-aware framework for converting off-the-shelf VLMs to MLA. Our approach features two core techniques: (1) a modality-adaptive partial-RoPE strategy that supports both traditional and multimodal settings by selectively masking nonessential dimensions, and (2) a modality-decoupled low-rank approximation method that independently compresses the visual and textual KV spaces. Furthermore, we introduce parameter-efficient fine-tuning to minimize adaptation cost and demonstrate that minimizing output activation error, rather than parameter distance, substantially reduces performance loss. Extensive experiments on three representative VLMs show that MHA2MLA-VLM restores original model performance with minimal supervised data, significantly reduces KV cache footprint, and integrates seamlessly with KV quantization. 

---
# Interactive Narrative Analytics: Bridging Computational Narrative Extraction and Human Sensemaking 

**Authors**: Brian Keith  

**Link**: [PDF](https://arxiv.org/pdf/2601.11459)  

**Abstract**: Information overload and misinformation create significant challenges in extracting meaningful narratives from large news collections. This paper defines the nascent field of Interactive Narrative Analytics (INA), which combines computational narrative extraction with interactive visual analytics to support sensemaking. INA approaches enable the interactive exploration of narrative structures through computational methods and visual interfaces that facilitate human interpretation. The field faces challenges in scalability, interactivity, knowledge integration, and evaluation standardization, yet offers promising opportunities across news analysis, intelligence, scientific literature exploration, and social media analysis. Through the combination of computational and human insight, INA addresses complex challenges in narrative sensemaking. 

---
# Isotropy-Optimized Contrastive Learning for Semantic Course Recommendation 

**Authors**: Ali Khreis, Anthony Nasr, Yusuf Hilal  

**Link**: [PDF](https://arxiv.org/pdf/2601.11427)  

**Abstract**: This paper presents a semantic course recommendation system for students using a self-supervised contrastive learning approach built upon BERT (Bidirectional Encoder Representations from Transformers). Traditional BERT embeddings suffer from anisotropic representation spaces, where course descriptions exhibit high cosine similarities regardless of semantic relevance. To address this limitation, we propose a contrastive learning framework with data augmentation and isotropy regularization that produces more discriminative embeddings. Our system processes student text queries and recommends Top-N relevant courses from a curated dataset of over 500 engineering courses across multiple faculties. Experimental results demonstrate that our fine-tuned model achieves improved embedding separation and more accurate course recommendations compared to vanilla BERT baselines. 

---
# PubMed-OCR: PMC Open Access OCR Annotations 

**Authors**: Hunter Heidenreich, Yosheb Getachew, Olivia Dinica, Ben Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2601.11425)  

**Abstract**: PubMed-OCR is an OCR-centric corpus of scientific articles derived from PubMed Central Open Access PDFs. Each page image is annotated with Google Cloud Vision and released in a compact JSON schema with word-, line-, and paragraph-level bounding boxes. The corpus spans 209.5K articles (1.5M pages; ~1.3B words) and supports layout-aware modeling, coordinate-grounded QA, and evaluation of OCR-dependent pipelines. We analyze corpus characteristics (e.g., journal coverage and detected layout features) and discuss limitations, including reliance on a single OCR engine and heuristic line reconstruction. We release the data and schema to facilitate downstream research and invite extensions. 

---
# AstroReason-Bench: Evaluating Unified Agentic Planning across Heterogeneous Space Planning Problems 

**Authors**: Weiyi Wang, Xinchi Chen, Jingjing Gong, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11354)  

**Abstract**: Recent advances in agentic Large Language Models (LLMs) have positioned them as generalist planners capable of reasoning and acting across diverse tasks. However, existing agent benchmarks largely focus on symbolic or weakly grounded environments, leaving their performance in physics-constrained real-world domains underexplored. We introduce AstroReason-Bench, a comprehensive benchmark for evaluating agentic planning in Space Planning Problems (SPP), a family of high-stakes problems with heterogeneous objectives, strict physical constraints, and long-horizon decision-making. AstroReason-Bench integrates multiple scheduling regimes, including ground station communication and agile Earth observation, and provides a unified agent-oriented interaction protocol. Evaluating on a range of state-of-the-art open- and closed-source agentic LLM systems, we find that current agents substantially underperform specialized solvers, highlighting key limitations of generalist planning under realistic constraints. AstroReason-Bench offers a challenging and diagnostic testbed for future agentic research. 

---
# Unlocking the Potentials of Retrieval-Augmented Generation for Diffusion Language Models 

**Authors**: Chuanyue Yu, Jiahui Wang, Yuhan Li, Heng Chang, Ge Lan, Qingyun Sun, Jia Li, Jianxin Li, Ziwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11342)  

**Abstract**: Diffusion Language Models (DLMs) have recently demonstrated remarkable capabilities in natural language processing tasks. However, the potential of Retrieval-Augmented Generation (RAG), which shows great successes for enhancing large language models (LLMs), has not been well explored, due to the fundamental difference between LLM and DLM decoding. To fill this critical gap, we systematically test the performance of DLMs within the RAG framework. Our findings reveal that DLMs coupled with RAG show promising potentials with stronger dependency on contextual information, but suffer from limited generation precision. We identify a key underlying issue: Response Semantic Drift (RSD), where the generated answer progressively deviates from the query's original semantics, leading to low precision content. We trace this problem to the denoising strategies in DLMs, which fail to maintain semantic alignment with the query throughout the iterative denoising process. To address this, we propose Semantic-Preserving REtrieval-Augmented Diffusion (SPREAD), a novel framework that introduces a query-relevance-guided denoising strategy. By actively guiding the denoising trajectory, SPREAD ensures the generation remains anchored to the query's semantics and effectively suppresses drift. Experimental results demonstrate that SPREAD significantly enhances the precision and effectively mitigates RSD of generated answers within the RAG framework. 

---
# Knowledge is Not Enough: Injecting RL Skills for Continual Adaptation 

**Authors**: Pingzhi Tang, Yiding Wang, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11258)  

**Abstract**: Large Language Models (LLMs) face the "knowledge cutoff" challenge, where their frozen parametric memory prevents direct internalization of new information. While Supervised Fine-Tuning (SFT) is commonly used to update model knowledge, it often updates factual content without reliably improving the model's ability to use the newly incorporated information for question answering or decision-making. Reinforcement Learning (RL) is essential for acquiring reasoning skills; however, its high computational cost makes it impractical for efficient online adaptation. We empirically observe that the parameter updates induced by SFT and RL are nearly orthogonal. Based on this observation, we propose Parametric Skill Transfer (PaST), a framework that supports modular skill transfer for efficient and effective knowledge adaptation. By extracting a domain-agnostic Skill Vector from a source domain, we can linearly inject knowledge manipulation skills into a target model after it has undergone lightweight SFT on new data. Experiments on knowledge-incorporation QA (SQuAD, LooGLE) and agentic tool-use benchmarks (ToolBench) demonstrate the effectiveness of our method. On SQuAD, PaST outperforms the state-of-the-art self-editing SFT baseline by up to 9.9 points. PaST further scales to long-context QA on LooGLE with an 8.0-point absolute accuracy gain, and improves zero-shot ToolBench success rates by +10.3 points on average with consistent gains across tool categories, indicating strong scalability and cross-domain transferability of the Skill Vector. 

---
# TANDEM: Temporal-Aware Neural Detection for Multimodal Hate Speech 

**Authors**: Girish A. Koushik, Helen Treharne, Diptesh Kanojia  

**Link**: [PDF](https://arxiv.org/pdf/2601.11178)  

**Abstract**: Social media platforms are increasingly dominated by long-form multimodal content, where harmful narratives are constructed through a complex interplay of audio, visual, and textual cues. While automated systems can flag hate speech with high accuracy, they often function as "black boxes" that fail to provide the granular, interpretable evidence, such as precise timestamps and target identities, required for effective human-in-the-loop moderation. In this work, we introduce TANDEM, a unified framework that transforms audio-visual hate detection from a binary classification task into a structured reasoning problem. Our approach employs a novel tandem reinforcement learning strategy where vision-language and audio-language models optimize each other through self-constrained cross-modal context, stabilizing reasoning over extended temporal sequences without requiring dense frame-level supervision. Experiments across three benchmark datasets demonstrate that TANDEM significantly outperforms zero-shot and context-augmented baselines, achieving 0.73 F1 in target identification on HateMM (a 30% improvement over state-of-the-art) while maintaining precise temporal grounding. We further observe that while binary detection is robust, differentiating between offensive and hateful content remains challenging in multi-class settings due to inherent label ambiguity and dataset imbalance. More broadly, our findings suggest that structured, interpretable alignment is achievable even in complex multimodal settings, offering a blueprint for the next generation of transparent and actionable online safety moderation tools. 

---
# FlashLabs Chroma 1.0: A Real-Time End-to-End Spoken Dialogue Model with Personalized Voice Cloning 

**Authors**: Tanyu Chen, Tairan Chen, Kai Shen, Zhenghua Bao, Zhihui Zhang, Man Yuan, Yi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2601.11141)  

**Abstract**: Recent end-to-end spoken dialogue systems leverage speech tokenizers and neural audio codecs to enable LLMs to operate directly on discrete speech representations. However, these models often exhibit limited speaker identity preservation, hindering personalized voice interaction. In this work, we present Chroma 1.0, the first open-source, real-time, end-to-end spoken dialogue model that achieves both low-latency interaction and high-fidelity personalized voice cloning. Chroma achieves sub-second end-to-end latency through an interleaved text-audio token schedule (1:2) that supports streaming generation, while maintaining high-quality personalized voice synthesis across multi-turn conversations. Our experimental results demonstrate that Chroma achieves a 10.96% relative improvement in speaker similarity over the human baseline, with a Real-Time Factor (RTF) of 0.43, while maintaining strong reasoning and dialogue capabilities. Our code and models are publicly available at this https URL and this https URL . 

---
# ABC-Bench: Benchmarking Agentic Backend Coding in Real-World Development 

**Authors**: Jie Yang, Honglin Guo, Li Ji, Jiazheng Zhou, Rui Zheng, Zhikai Lei, Shuo Zhang, Zhiheng Xi, Shichun Liu, Yuxin Wang, Bo Wang, Yining Zheng, Tao Gui, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11077)  

**Abstract**: The evolution of Large Language Models (LLMs) into autonomous agents has expanded the scope of AI coding from localized code generation to complex, repository-level, and execution-driven problem solving. However, current benchmarks predominantly evaluate code logic in static contexts, neglecting the dynamic, full-process requirements of real-world engineering, particularly in backend development which demands rigorous environment configuration and service deployment. To address this gap, we introduce ABC-Bench, a benchmark explicitly designed to evaluate agentic backend coding within a realistic, executable workflow. Using a scalable automated pipeline, we curated 224 practical tasks spanning 8 languages and 19 frameworks from open-source repositories. Distinct from previous evaluations, ABC-Bench require the agents to manage the entire development lifecycle from repository exploration to instantiating containerized services and pass the external end-to-end API tests. Our extensive evaluation reveals that even state-of-the-art models struggle to deliver reliable performance on these holistic tasks, highlighting a substantial disparity between current model capabilities and the demands of practical backend engineering. Our code is available at this https URL. 

---
# Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs 

**Authors**: Lecheng Yan, Ruizhe Li, Guanhua Chen, Qing Li, Jiahui Geng, Wenxi Li, Vincent Wang, Chris Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.11061)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is highly effective for enhancing LLM reasoning, yet recent evidence shows models like Qwen 2.5 achieve significant gains even with spurious or incorrect rewards. We investigate this phenomenon and identify a "Perplexity Paradox": spurious RLVR triggers a divergence where answer-token perplexity drops while prompt-side coherence degrades, suggesting the model is bypassing reasoning in favor of memorization. Using Path Patching, Logit Lens, JSD analysis, and Neural Differential Equations, we uncover a hidden Anchor-Adapter circuit that facilitates this shortcut. We localize a Functional Anchor in the middle layers (L18-20) that triggers the retrieval of memorized solutions, followed by Structural Adapters in later layers (L21+) that transform representations to accommodate the shortcut signal. Finally, we demonstrate that scaling specific MLP keys within this circuit allows for bidirectional causal steering-artificially amplifying or suppressing contamination-driven performance. Our results provide a mechanistic roadmap for identifying and mitigating data contamination in RLVR-tuned models. Code is available at this https URL. 

---
# SonicBench: Dissecting the Physical Perception Bottleneck in Large Audio Language Models 

**Authors**: Yirong Sun, Yanjun Chen, Xin Qiu, Gang Zhang, Hongyu Chen, Daokuan Wu, Chengming Li, Min Yang, Dawei Zhu, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11039)  

**Abstract**: Large Audio Language Models (LALMs) excel at semantic and paralinguistic tasks, yet their ability to perceive the fundamental physical attributes of audio such as pitch, loudness, and spatial location remains under-explored. To bridge this gap, we introduce SonicBench, a psychophysically grounded benchmark that systematically evaluates 12 core physical attributes across five perceptual dimensions. Unlike previous datasets, SonicBench uses a controllable generation toolbox to construct stimuli for two complementary paradigms: recognition (absolute judgment) and comparison (relative judgment). This design allows us to probe not only sensory precision but also relational reasoning capabilities, a domain where humans typically exhibit greater proficiency. Our evaluation reveals a substantial deficiency in LALMs' foundational auditory understanding; most models perform near random guessing and, contrary to human patterns, fail to show the expected advantage on comparison tasks. Furthermore, explicit reasoning yields minimal gains. However, our linear probing analysis demonstrates crucially that frozen audio encoders do successfully capture these physical cues (accuracy at least 60%), suggesting that the primary bottleneck lies in the alignment and decoding stages, where models fail to leverage the sensory signals they have already captured. 

---
# AdaMARP: An Adaptive Multi-Agent Interaction Framework for General Immersive Role-Playing 

**Authors**: Zhenhua Xu, Dongsheng Chen, Shuo Wang, Jian Li, Chengjie Wang, Meng Han, Yabiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11007)  

**Abstract**: LLM role-playing aims to portray arbitrary characters in interactive narratives, yet existing systems often suffer from limited immersion and adaptability. They typically under-model dynamic environmental information and assume largely static scenes and casts, offering insufficient support for multi-character orchestration, scene transitions, and on-the-fly character introduction. We propose an adaptive multi-agent role-playing framework, AdaMARP, featuring an immersive message format that interleaves [Thought], (Action), <Environment>, and Speech, together with an explicit Scene Manager that governs role-playing through discrete actions (init_scene, pick_speaker, switch_scene, add_role, end) accompanied by rationales. To train these capabilities, we construct AdaRPSet for the Actor Model and AdaSMSet for supervising orchestration decisions, and introduce AdaptiveBench for trajectory-level evaluation. Experiments across multiple backbones and model scales demonstrate consistent improvements: AdaRPSet enhances character consistency, environment grounding, and narrative coherence, with an 8B actor outperforming several commercial LLMs, while AdaSMSet enables smoother scene transitions and more natural role introductions, surpassing Claude Sonnet 4.5 using only a 14B LLM. 

---
# AJAR: Adaptive Jailbreak Architecture for Red-teaming 

**Authors**: Yipu Dou, Wang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10971)  

**Abstract**: As Large Language Models (LLMs) evolve from static chatbots into autonomous agents capable of tool execution, the landscape of AI safety is shifting from content moderation to action security. However, existing red-teaming frameworks remain bifurcated: they either focus on rigid, script-based text attacks or lack the architectural modularity to simulate complex, multi-turn agentic exploitations. In this paper, we introduce AJAR (Adaptive Jailbreak Architecture for Red-teaming), a proof-of-concept framework designed to bridge this gap through Protocol-driven Cognitive Orchestration. Built upon the robust runtime of Petri, AJAR leverages the Model Context Protocol (MCP) to decouple adversarial logic from the execution loop, encapsulating state-of-the-art algorithms like X-Teaming as standardized, plug-and-play services. We validate the architectural feasibility of AJAR through a controlled qualitative case study, demonstrating its ability to perform stateful backtracking within a tool-use environment. Furthermore, our preliminary exploration of the "Agentic Gap" reveals a complex safety dynamic: while tool usage introduces new injection vectors via code execution, the cognitive load of parameter formatting can inadvertently disrupt persona-based attacks. AJAR is open-sourced to facilitate the standardized, environment-aware evaluation of this emerging attack surface. The code and data are available at this https URL. 

---
# PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis 

**Authors**: K Lokesh, Abhirama Subramanyam Penamakuri, Uday Agarwal, Apoorva Challa, Shreya K Gowda, Somesh Gupta, Anand Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2601.10945)  

**Abstract**: Traditionally, AI research in medical diagnosis has largely centered on image analysis. While this has led to notable advancements, the absence of patient-reported symptoms continues to hinder diagnostic accuracy. To address this, we propose a Pre-Consultation Dialogue Framework (PCDF) that mimics real-world diagnostic procedures, where doctors iteratively query patients before reaching a conclusion. Specifically, we simulate diagnostic dialogues between two vision-language models (VLMs): a DocVLM, which generates follow-up questions based on the image and dialogue history, and a PatientVLM, which responds using a symptom profile derived from the ground-truth diagnosis. We additionally conducted a small-scale clinical validation of the synthetic symptoms generated by our framework, with licensed clinicians confirming their clinical relevance, symptom coverage, and overall realism. These findings indicate that the resulting DocVLM-PatientVLM interactions form coherent, multi-turn consultations paired with images and diagnoses, which we then use to fine-tune the DocVLM. This dialogue-based supervision leads to substantial gains over image-only training, highlighting the value of realistic symptom elicitation for diagnosis. 

---
# Towards Reliable ML Feature Engineering via Planning in Constrained-Topology of LLM Agents 

**Authors**: Himanshu Thakur, Anusha Kamath, Anurag Muthyala, Dhwani Sanmukhani, Smruthi Mukund, Jay Katukuri  

**Link**: [PDF](https://arxiv.org/pdf/2601.10820)  

**Abstract**: Recent advances in code generation models have unlocked unprecedented opportunities for automating feature engineering, yet their adoption in real-world ML teams remains constrained by critical challenges: (i) the scarcity of datasets capturing the iterative and complex coding processes of production-level feature engineering, (ii) limited integration and personalization of widely used coding agents, such as CoPilot and Devin, with a team's unique tools, codebases, workflows, and practices, and (iii) suboptimal human-AI collaboration due to poorly timed or insufficient feedback. We address these challenges with a planner-guided, constrained-topology multi-agent framework that generates code for repositories in a multi-step fashion. The LLM-powered planner leverages a team's environment, represented as a graph, to orchestrate calls to available agents, generate context-aware prompts, and use downstream failures to retroactively correct upstream artifacts. It can request human intervention at critical steps, ensuring generated code is reliable, maintainable, and aligned with team expectations. On a novel in-house dataset, our approach achieves 38% and 150% improvement in the evaluation metric over manually crafted and unplanned workflows respectively. In practice, when building features for recommendation models serving over 120 million users, our approach has delivered real-world impact by reducing feature engineering cycles from three weeks to a single day. 

---
# Building AI Agents to Improve Job Referral Requests to Strangers 

**Authors**: Ross Chu, Yuting Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10726)  

**Abstract**: This paper develops AI agents that help job seekers write effective requests for job referrals in a professional online community. The basic workflow consists of an improver agent that rewrites the referral request and an evaluator agent that measures the quality of revisions using a model trained to predict the probability of receiving referrals from other users. Revisions suggested by the LLM (large language model) increase predicted success rates for weaker requests while reducing them for stronger requests. Enhancing the LLM with Retrieval-Augmented Generation (RAG) prevents edits that worsen stronger requests while it amplifies improvements for weaker requests. Overall, using LLM revisions with RAG increases the predicted success rate for weaker requests by 14\% without degrading performance on stronger requests. Although improvements in model-predicted success do not guarantee more referrals in the real world, they provide low-cost signals for promising features before running higher-stakes experiments on real users. 

---
# Do You Trust Me? Cognitive-Affective Signatures of Trustworthiness in Large Language Models 

**Authors**: Gerard Yeo, Svetlana Churina, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2601.10719)  

**Abstract**: Perceived trustworthiness underpins how users navigate online information, yet it remains unclear whether large language models (LLMs),increasingly embedded in search, recommendation, and conversational systems, represent this construct in psychologically coherent ways. We analyze how instruction-tuned LLMs (Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B) encode perceived trustworthiness in web-like narratives using the PEACE-Reviews dataset annotated for cognitive appraisals, emotions, and behavioral intentions. Across models, systematic layer- and head-level activation differences distinguish high- from low-trust texts, revealing that trust cues are implicitly encoded during pretraining. Probing analyses show linearly de-codable trust signals and fine-tuning effects that refine rather than restructure these representations. Strongest associations emerge with appraisals of fairness, certainty, and accountability-self -- dimensions central to human trust formation online. These findings demonstrate that modern LLMs internalize psychologically grounded trust signals without explicit supervision, offering a representational foundation for designing credible, transparent, and trust-worthy AI systems in the web ecosystem. Code and appendix are available at: this https URL. 

---
# Japanese AI Agent System on Human Papillomavirus Vaccination: System Design 

**Authors**: Junyu Liu, Siwen Yang, Dexiu Ma, Qian Niu, Zequn Zhang, Momoko Nagai-Tanima, Tomoki Aoyama  

**Link**: [PDF](https://arxiv.org/pdf/2601.10718)  

**Abstract**: Human papillomavirus (HPV) vaccine hesitancy poses significant public health challenges, particularly in Japan where proactive vaccination recommendations were suspended from 2013 to 2021. The resulting information gap is exacerbated by misinformation on social media, and traditional ways cannot simultaneously address individual queries while monitoring population-level discourse. This study aimed to develop a dual-purpose AI agent system that provides verified HPV vaccine information through a conversational interface while generating analytical reports for medical institutions based on user interactions and social media. We implemented a system comprising: a vector database integrating academic papers, government sources, news media, and social media; a Retrieval-Augmented Generation chatbot using ReAct agent architecture with multi-tool orchestration across five knowledge sources; and an automated report generation system with modules for news analysis, research synthesis, social media sentiment analysis, and user interaction pattern identification. Performance was assessed using a 0-5 scoring scale. For single-turn evaluation, the chatbot achieved mean scores of 4.83 for relevance, 4.89 for routing, 4.50 for reference quality, 4.90 for correctness, and 4.88 for professional identity (overall 4.80). Multi-turn evaluation yielded higher scores: context retention 4.94, topic coherence 5.00, and overall 4.98. The report generation system achieved completeness 4.00-5.00, correctness 4.00-5.00, and helpfulness 3.67-5.00, with reference validity 5.00 across all periods. This study demonstrates the feasibility of an integrated AI agent system for bidirectional HPV vaccine communication. The architecture enables verified information delivery with source attribution while providing systematic public discourse analysis, with a transferable framework for adaptation to other medical contexts. 

---
# MPCI-Bench: A Benchmark for Multimodal Pairwise Contextual Integrity Evaluation of Language Model Agents 

**Authors**: Shouju Wang, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08235)  

**Abstract**: As language-model agents evolve from passive chatbots into proactive assistants that handle personal data, evaluating their adherence to social norms becomes increasingly critical, often through the lens of Contextual Integrity (CI). However, existing CI benchmarks are largely text-centric and primarily emphasize negative refusal scenarios, overlooking multimodal privacy risks and the fundamental trade-off between privacy and utility. In this paper, we introduce MPCI-Bench, the first Multimodal Pairwise Contextual Integrity benchmark for evaluating privacy behavior in agentic settings. MPCI-Bench consists of paired positive and negative instances derived from the same visual source and instantiated across three tiers: normative Seed judgments, context-rich Story reasoning, and executable agent action Traces. Data quality is ensured through a Tri-Principle Iterative Refinement pipeline. Evaluations of state-of-the-art multimodal models reveal systematic failures to balance privacy and utility and a pronounced modality leakage gap, where sensitive visual information is leaked more frequently than textual information. We will open-source MPCI-Bench to facilitate future research on agentic CI. 

---
# Generative AI Purpose-built for Social and Mental Health: A Real-World Pilot 

**Authors**: Thomas D. Hull, Lizhe Zhang, Patricia A. Arean, Matteo Malgaroli  

**Link**: [PDF](https://arxiv.org/pdf/2511.11689)  

**Abstract**: Generative artificial intelligence (GAI) chatbots built for mental health could deliver safe, personalized, and scalable mental health support. We evaluate a foundation model designed for mental health. Adults completed mental health measures while engaging with the chatbot between May 15, 2025 and September 15, 2025. Users completed an opt-in consent, demographic information, mental health symptoms, social connection, and self-identified goals. Measures were repeated every two weeks up to 6 weeks, and a final follow-up at 10 weeks. Analyses included effect sizes, and growth mixture models to identify participant groups and their characteristic engagement, severity, and demographic factors. Users demonstrated significant reductions in PHQ-9 and GAD-7 that were sustained at follow-up. Significant improvements in Hope, Behavioral Activation, Social Interaction, Loneliness, and Perceived Social Support were observed throughout and maintained at 10 week follow-up. Engagement was high and predicted outcomes. Working alliance was comparable to traditional care and predicted outcomes. Automated safety guardrails functioned as designed, with 76 sessions flagged for risk and all handled according to escalation policies. This single arm naturalistic observational study provides initial evidence that a GAI foundation model for mental health can deliver accessible, engaging, effective, and safe mental health support. These results lend support to findings from early randomized designs and offer promise for future study of mental health GAI in real world settings. 

---
