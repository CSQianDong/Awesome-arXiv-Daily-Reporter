# Less is More: Benchmarking LLM Based Recommendation Agents 

**Authors**: Kargi Chauhan, Mahalakshmi Venkateswarlu  

**Link**: [PDF](https://arxiv.org/pdf/2601.20316)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed for personalized product recommendations, with practitioners commonly assuming that longer user purchase histories lead to better predictions. We challenge this assumption through a systematic benchmark of four state of the art LLMs GPT-4o-mini, DeepSeek-V3, Qwen2.5-72B, and Gemini 2.5 Flash across context lengths ranging from 5 to 50 items using the REGEN dataset.
Surprisingly, our experiments with 50 users in a within subject design reveal no significant quality improvement with increased context length. Quality scores remain flat across all conditions (0.17--0.23). Our findings have significant practical implications: practitioners can reduce inference costs by approximately 88\% by using context (5--10 items) instead of longer histories (50 items), without sacrificing recommendation quality. We also analyze latency patterns across providers and find model specific behaviors that inform deployment decisions. This work challenges the existing ``more context is better'' paradigm and provides actionable guidelines for cost effective LLM based recommendation systems. 

---
# When Vision Meets Texts in Listwise Reranking 

**Authors**: Hongyi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2601.20623)  

**Abstract**: Recent advancements in information retrieval have highlighted the potential of integrating visual and textual information, yet effective reranking for image-text documents remains challenging due to the modality gap and scarcity of aligned datasets. Meanwhile, existing approaches often rely on large models (7B to 32B parameters) with reasoning-based distillation, incurring unnecessary computational overhead while primarily focusing on textual modalities. In this paper, we propose Rank-Nexus, a multimodal image-text document reranker that performs listwise qualitative reranking on retrieved lists incorporating both images and texts. To bridge the modality gap, we introduce a progressive cross-modal training strategy. We first train modalities separately: leveraging abundant text reranking data, we distill knowledge into the text branch. For images, where data is scarce, we construct distilled pairs from multimodal large language model (MLLM) captions on image retrieval benchmarks. Subsequently, we distill a joint image-text reranking dataset. Rank-Nexus achieves outstanding performance on text reranking benchmarks (TREC, BEIR) and the challenging image reranking benchmark (INQUIRE, MMDocIR), using only a lightweight 2B pretrained visual-language model. This efficient design ensures strong generalization across diverse multimodal scenarios without excessive parameters or reasoning overhead. 

---
# Persona Prompting as a Lens on LLM Social Reasoning 

**Authors**: Jing Yang, Moritz Hechtbauer, Elisabeth Khalilov, Evelyn Luise Brinkmann, Vera Schmitt, Nils Feldhus  

**Link**: [PDF](https://arxiv.org/pdf/2601.20757)  

**Abstract**: For socially sensitive tasks like hate speech detection, the quality of explanations from Large Language Models (LLMs) is crucial for factors like user trust and model alignment. While Persona prompting (PP) is increasingly used as a way to steer model towards user-specific generation, its effect on model rationales remains underexplored. We investigate how LLM-generated rationales vary when conditioned on different simulated demographic personas. Using datasets annotated with word-level rationales, we measure agreement with human annotations from different demographic groups, and assess the impact of PP on model bias and human alignment. Our evaluation across three LLMs results reveals three key findings: (1) PP improving classification on the most subjective task (hate speech) but degrading rationale quality. (2) Simulated personas fail to align with their real-world demographic counterparts, and high inter-persona agreement shows models are resistant to significant steering. (3) Models exhibit consistent demographic biases and a strong tendency to over-flag content as harmful, regardless of PP. Our findings reveal a critical trade-off: while PP can improve classification in socially-sensitive tasks, it often comes at the cost of rationale quality and fails to mitigate underlying biases, urging caution in its application. 

---
# Harnessing Large Language Models for Precision Querying and Retrieval-Augmented Knowledge Extraction in Clinical Data Science 

**Authors**: Juan Jose Rubio Jan, Jack Wu, Julia Ive  

**Link**: [PDF](https://arxiv.org/pdf/2601.20674)  

**Abstract**: This study applies Large Language Models (LLMs) to two foundational Electronic Health Record (EHR) data science tasks: structured data querying (using programmatic languages, Python/Pandas) and information extraction from unstructured clinical text via a Retrieval Augmented Generation (RAG) pipeline. We test the ability of LLMs to interact accurately with large structured datasets for analytics and the reliability of LLMs in extracting semantically correct information from free text health records when supported by RAG. To this end, we presented a flexible evaluation framework that automatically generates synthetic question and answer pairs tailored to the characteristics of each dataset or task. Experiments were conducted on a curated subset of MIMIC III, (four structured tables and one clinical note type), using a mix of locally hosted and API-based LLMs. Evaluation combined exact-match metrics, semantic similarity, and human judgment. Our findings demonstrate the potential of LLMs to support precise querying and accurate information extraction in clinical workflows. 

---
# P2S: Probabilistic Process Supervision for General-Domain Reasoning Question Answering 

**Authors**: Wenlin Zhong, Chengyuan Liu, Yiquan Wu, Bovin Tan, Changlong Sun, Yi Wang, Xiaozhong Liu, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2601.20649)  

**Abstract**: While reinforcement learning with verifiable rewards (RLVR) has advanced LLM reasoning in structured domains like mathematics and programming, its application to general-domain reasoning tasks remains challenging due to the absence of verifiable reward signals. To this end, methods like Reinforcement Learning with Reference Probability Reward (RLPR) have emerged, leveraging the probability of generating the final answer as a reward signal. However, these outcome-focused approaches neglect crucial step-by-step supervision of the reasoning process itself. To address this gap, we introduce Probabilistic Process Supervision (P2S), a novel self-supervision framework that provides fine-grained process rewards without requiring a separate reward model or human-annotated reasoning steps. During reinforcement learning, P2S synthesizes and filters a high-quality reference reasoning chain (gold-CoT). The core of our method is to calculate a Path Faithfulness Reward (PFR) for each reasoning step, which is derived from the conditional probability of generating the gold-CoT's suffix, given the model's current reasoning prefix. Crucially, this PFR can be flexibly integrated with any outcome-based reward, directly tackling the reward sparsity problem by providing dense guidance. Extensive experiments on reading comprehension and medical Question Answering benchmarks show that P2S significantly outperforms strong baselines. 

---
# A Dialectic Pipeline for Improving LLM Robustness 

**Authors**: Sara Candussio  

**Link**: [PDF](https://arxiv.org/pdf/2601.20659)  

**Abstract**: Assessing ways in which Language Models can reduce their hallucinations and improve the outputs' quality is crucial to ensure their large-scale use.
However, methods such as fine-tuning on domain-specific data or the training of a separate \textit{ad hoc} verifier require demanding computational resources (not feasible for many user applications) and constrain the models to specific fields of knowledge.
In this thesis, we propose a dialectic pipeline that preserves LLMs' generalization abilities while improving the quality of its answer via self-dialogue, enabling it to reflect upon and correct tentative wrong answers.
We experimented with different pipeline settings, testing our proposed method on different datasets and on different families of models. All the pipeline stages are enriched with the relevant context (in an oracle-RAG setting) and a study on the impact of its summarization or its filtering is conducted.
We find that our proposed dialectic pipeline is able to outperform by significative margins the standard model answers and that it consistently achieves higher performances than Chain-of-Thought only prompting. 

---
# When Flores Bloomz Wrong: Cross-Direction Contamination in Machine Translation Evaluation 

**Authors**: David Tan, Pinzhen Chen, Josef van Genabith, Koel Dutta Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2601.20858)  

**Abstract**: Large language models (LLMs) can be benchmark-contaminated, resulting in inflated scores that mask memorization as generalization, and in multilingual settings, this memorization can even transfer to "uncontaminated" languages. Using the FLORES-200 translation benchmark as a diagnostic, we study two 7-8B instruction-tuned multilingual LLMs: Bloomz, which was trained on FLORES, and Llama as an uncontaminated control. We confirm Bloomz's FLORES contamination and demonstrate that machine translation contamination can be cross-directional, artificially boosting performance in unseen translation directions due to target-side memorization. Further analysis shows that recall of memorized references often persists despite various source-side perturbation efforts like paraphrasing and named entity replacement. However, replacing named entities leads to a consistent decrease in BLEU, suggesting an effective probing method for memorization in contaminated models. 

---
# Linear representations in language models can change dramatically over a conversation 

**Authors**: Andrew Kyle Lampinen, Yuxuan Li, Eghbal Hosseini, Sangnie Bhardwaj, Murray Shanahan  

**Link**: [PDF](https://arxiv.org/pdf/2601.20834)  

**Abstract**: Language model representations often contain linear directions that correspond to high-level concepts. Here, we study the dynamics of these representations: how representations evolve along these dimensions within the context of (simulated) conversations. We find that linear representations can change dramatically over a conversation; for example, information that is represented as factual at the beginning of a conversation can be represented as non-factual at the end and vice versa. These changes are content-dependent; while representations of conversation-relevant information may change, generic information is generally preserved. These changes are robust even for dimensions that disentangle factuality from more superficial response patterns, and occur across different model families and layers of the model. These representation changes do not require on-policy conversations; even replaying a conversation script written by an entirely different model can produce similar changes. However, adaptation is much weaker from simply having a sci-fi story in context that is framed more explicitly as such. We also show that steering along a representational direction can have dramatically different effects at different points in a conversation. These results are consistent with the idea that representations may evolve in response to the model playing a particular role that is cued by a conversation. Our findings may pose challenges for interpretability and steering -- in particular, they imply that it may be misleading to use static interpretations of features or directions, or probes that assume a particular range of features consistently corresponds to a particular ground-truth value. However, these types of representational dynamics also point to exciting new research directions for understanding how models adapt to context. 

---
# Beyond Divergent Creativity: A Human-Based Evaluation of Creativity in Large Language Models 

**Authors**: Kumiko Nakajima, Jan Zuiderveld, Sandro Pezzelle  

**Link**: [PDF](https://arxiv.org/pdf/2601.20546)  

**Abstract**: Large language models (LLMs) are increasingly used in verbal creative tasks. However, previous assessments of the creative capabilities of LLMs remain weakly grounded in human creativity theory and are thus hard to interpret. The widely used Divergent Association Task (DAT) focuses on novelty, ignoring appropriateness, a core component of creativity. We evaluate a range of state-of-the-art LLMs on DAT and show that their scores on the task are lower than those of two baselines that do not possess any creative abilities, undermining its validity for model evaluation. Grounded in human creativity theory, which defines creativity as the combination of novelty and appropriateness, we introduce Conditional Divergent Association Task (CDAT). CDAT evaluates novelty conditional on contextual appropriateness, separating noise from creativity better than DAT, while remaining simple and objective. Under CDAT, smaller model families often show the most creativity, whereas advanced families favor appropriateness at lower novelty. We hypothesize that training and alignment likely shift models along this frontier, making outputs more appropriate but less creative. We release the dataset and code. 

---
# Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents 

**Authors**: Qihao Wang, Yue Hu, Mingzhe Lu, Jiayue Wu, Yanbing Liu, Yuanmin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.20412)  

**Abstract**: The ability of Large Language Models (LLMs) to use external tools unlocks powerful real-world interactions, making rigorous evaluation essential. However, current benchmarks primarily report final accuracy, revealing what models can do but obscuring the cognitive bottlenecks that define their true capability boundaries. To move from simple performance scoring to a diagnostic tool, we introduce a framework grounded in Cognitive Load Theory. Our framework deconstructs task complexity into two quantifiable components: Intrinsic Load, the inherent structural complexity of the solution path, formalized with a novel Tool Interaction Graph; and Extraneous Load, the difficulty arising from ambiguous task presentation. To enable controlled experiments, we construct ToolLoad-Bench, the first benchmark with parametrically adjustable cognitive load. Our evaluation reveals distinct performance cliffs as cognitive load increases, allowing us to precisely map each model's capability boundary. We validate that our framework's predictions are highly calibrated with empirical results, establishing a principled methodology for understanding an agent's limits and a practical foundation for building more efficient systems. 

---
# Can We Improve Educational Diagram Generation with In-Context Examples? Not if a Hallucination Spoils the Bunch 

**Authors**: Evanfiya Logacheva, Arto Hellas, Tsvetomila Mihaylova, Juha Sorva, Ava Heinonen, Juho Leinonen  

**Link**: [PDF](https://arxiv.org/pdf/2601.20476)  

**Abstract**: Generative artificial intelligence (AI) has found a widespread use in computing education; at the same time, quality of generated materials raises concerns among educators and students. This study addresses this issue by introducing a novel method for diagram code generation with in-context examples based on the Rhetorical Structure Theory (RST), which aims to improve diagram generation by aligning models' output with user expectations. Our approach is evaluated by computer science educators, who assessed 150 diagrams generated with large language models (LLMs) for logical organization, connectivity, layout aesthetic, and AI hallucination. The assessment dataset is additionally investigated for its utility in automated diagram evaluation. The preliminary results suggest that our method decreases the rate of factual hallucination and improves diagram faithfulness to provided context; however, due to LLMs' stochasticity, the quality of the generated diagrams varies. Additionally, we present an in-depth analysis and discussion on the connection between AI hallucination and the quality of generated diagrams, which reveals that text contexts of higher complexity lead to higher rates of hallucination and LLMs often fail to detect mistakes in their output. 

---
# PsychePass: Calibrating LLM Therapeutic Competence via Trajectory-Anchored Tournaments 

**Authors**: Zhuang Chen, Dazhen Wan, Zhangkai Zheng, Guanqun Bi, Xiyao Xiao, Binghang Li, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.20330)  

**Abstract**: While large language models show promise in mental healthcare, evaluating their therapeutic competence remains challenging due to the unstructured and longitudinal nature of counseling. We argue that current evaluation paradigms suffer from an unanchored defect, leading to two forms of instability: process drift, where unsteered client simulation wanders away from specific counseling goals, and standard drift, where static pointwise scoring lacks the stability for reliable judgment. To address this, we introduce Ps, a unified framework that calibrates the therapeutic competence of LLMs via trajectory-anchored tournaments. We first anchor the interaction trajectory in simulation, where clients precisely control the fluid consultation process to probe multifaceted capabilities. We then anchor the battle trajectory in judgments through an efficient Swiss-system tournament, utilizing dynamic pairwise battles to yield robust Elo ratings. Beyond ranking, we demonstrate that tournament trajectories can be transformed into credible reward signals, enabling on-policy reinforcement learning to enhance LLMs' performance. Extensive experiments validate the effectiveness of PsychePass and its strong consistency with human expert judgments. 

---
# QueerGen: How LLMs Reflect Societal Norms on Gender and Sexuality in Sentence Completion Tasks 

**Authors**: Mae Sosto, Delfina Sol Martinez Pandiani, Laura Hollink  

**Link**: [PDF](https://arxiv.org/pdf/2601.20731)  

**Abstract**: This paper examines how Large Language Models (LLMs) reproduce societal norms, particularly heterocisnormativity, and how these norms translate into measurable biases in their text generations. We investigate whether explicit information about a subject's gender or sexuality influences LLM responses across three subject categories: queer-marked, non-queer-marked, and the normalized "unmarked" category. Representational imbalances are operationalized as measurable differences in English sentence completions across four dimensions: sentiment, regard, toxicity, and prediction diversity. Our findings show that Masked Language Models (MLMs) produce the least favorable sentiment, higher toxicity, and more negative regard for queer-marked subjects. Autoregressive Language Models (ARLMs) partially mitigate these patterns, while closed-access ARLMs tend to produce more harmful outputs for unmarked subjects. Results suggest that LLMs reproduce normative social assumptions, though the form and degree of bias depend strongly on specific model characteristics, which may redistribute, but not eliminate, representational harms. 

---
# SpeechMapper: Speech-to-text Embedding Projector for LLMs 

**Authors**: Biswesh Mohapatra, Marcely Zanon Boito, Ioan Calapodescu  

**Link**: [PDF](https://arxiv.org/pdf/2601.20417)  

**Abstract**: Current speech LLMs bridge speech foundation models to LLMs using projection layers, training all of these components on speech instruction data. This strategy is computationally intensive and susceptible to task and prompt overfitting. We present SpeechMapper, a cost-efficient speech-to-LLM-embedding training approach that mitigates overfitting, enabling more robust and generalizable models. Our model is first pretrained without the LLM on inexpensive hardware, and then efficiently attached to the target LLM via a brief 1K-step instruction tuning (IT) stage. Through experiments on speech translation and spoken question answering, we demonstrate the versatility of SpeechMapper's pretrained block, presenting results for both task-agnostic IT, an ASR-based adaptation strategy that does not train in the target task, and task-specific IT. In task-agnostic settings, Speechmapper rivals the best instruction-following speech LLM from IWSLT25, despite never being trained on these tasks, while in task-specific settings, it outperforms this model across many datasets, despite requiring less data and compute. Overall, SpeechMapper offers a practical and scalable approach for efficient, generalizable speech-LLM integration without large-scale IT. 

---
# Automated Benchmark Generation from Domain Guidelines Informed by Bloom's Taxonomy 

**Authors**: Si Chen, Le Huy Khiem, Annalisa Szymanski, Ronald Metoyer, Ting Hua, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2601.20253)  

**Abstract**: Open-ended question answering (QA) evaluates a model's ability to perform contextualized reasoning beyond factual recall. This challenge is especially acute in practice-based domains, where knowledge is procedural and grounded in professional judgment, while most existing LLM benchmarks depend on pre-existing human exam datasets that are often unavailable in such settings. We introduce a framework for automated benchmark generation from expert-authored guidelines informed by Bloom's Taxonomy. It converts expert practices into implicit violation-based scenarios and expands them into auto-graded multiple-choice questions (MCQs) and multi-turn dialogues across four cognitive levels, enabling deterministic, reproducible, and scalable evaluation. Applied to three applied domains: teaching, dietetics, and caregiving, we find differences between model and human-like reasoning: LLMs sometimes perform relatively better on higher-order reasoning (Analyze) but fail more frequently on lower-level items (Remember). We produce large-scale, psychometrically informed benchmarks that surface these non-intuitive model behaviors and enable evaluation of contextualized reasoning in real-world settings. 

---
# Unit-Based Agent for Semi-Cascaded Full-Duplex Dialogue Systems 

**Authors**: Haoyuan Yu, Yuxuan Chen, Minjie Cai  

**Link**: [PDF](https://arxiv.org/pdf/2601.20230)  

**Abstract**: Full-duplex voice interaction is crucial for natural human computer interaction. We present a framework that decomposes complex dialogue into minimal conversational units, enabling the system to process each unit independently and predict when to transit to the next. This framework is instantiated as a semi-cascaded full-duplex dialogue system built around a multimodal large language model, supported by auxiliary modules such as voice activity detection (VAD) and text-to-speech (TTS) synthesis. The resulting system operates in a train-free, plug-and-play manner. Experiments on the HumDial dataset demonstrate the effectiveness of our framework, which ranks second among all teams on the test set of the Human-like Spoken Dialogue Systems Challenge (Track 2: Full-Duplex Interaction). Code is available at the GitHub repository this https URL. 

---
# VERGE: Formal Refinement and Guidance Engine for Verifiable LLM Reasoning 

**Authors**: Vikash Singh, Darion Cassel, Nathaniel Weir, Nick Feng, Sam Bayless  

**Link**: [PDF](https://arxiv.org/pdf/2601.20055)  

**Abstract**: Despite the syntactic fluency of Large Language Models (LLMs), ensuring their logical correctness in high-stakes domains remains a fundamental challenge. We present a neurosymbolic framework that combines LLMs with SMT solvers to produce verification-guided answers through iterative refinement. Our approach decomposes LLM outputs into atomic claims, autoformalizes them into first-order logic, and verifies their logical consistency using automated theorem proving. We introduce three key innovations: (1) multi-model consensus via formal semantic equivalence checking to ensure logic-level alignment between candidates, eliminating the syntactic bias of surface-form metrics, (2) semantic routing that directs different claim types to appropriate verification strategies: symbolic solvers for logical claims and LLM ensembles for commonsense reasoning, and (3) precise logical error localization via Minimal Correction Subsets (MCS), which pinpoint the exact subset of claims to revise, transforming binary failure signals into actionable feedback. Our framework classifies claims by their logical status and aggregates multiple verification signals into a unified score with variance-based penalty. The system iteratively refines answers using structured feedback until acceptance criteria are met or convergence is achieved. This hybrid approach delivers formal guarantees where possible and consensus verification elsewhere, advancing trustworthy AI. With the GPT-OSS-120B model, VERGE demonstrates an average performance uplift of 18.7% at convergence across a set of reasoning benchmarks compared to single-pass approaches. 

---
# LinguaMap: Which Layers of LLMs Speak Your Language and How to Tune Them? 

**Authors**: J. Ben Tamo, Daniel Carlander-Reuterfelt, Jonathan Rubin, Dezhi Hong, Mingxian Wang, Oleg Poliannikov  

**Link**: [PDF](https://arxiv.org/pdf/2601.20009)  

**Abstract**: Despite multilingual pretraining, large language models often struggle with non-English tasks, particularly in language control, the ability to respond in the intended language. We identify and characterize two key failure modes: the multilingual transfer bottleneck (correct language, incorrect task response) and the language consistency bottleneck (correct task response, wrong language). To systematically surface these issues, we design a four-scenario evaluation protocol spanning MMLU, MGSM, and XQuAD benchmarks. To probe these issues with interpretability, we extend logit lens analysis to track language probabilities layer by layer and compute cross-lingual semantic similarity of hidden states. The results reveal a three-phase internal structure: early layers align inputs into a shared semantic space, middle layers perform task reasoning, and late layers drive language-specific generation. Guided by these insights, we introduce selective fine-tuning of only the final layers responsible for language control. On Qwen-3-32B and Bloom-7.1B, this method achieves over 98 percent language consistency across six languages while fine-tuning only 3-5 percent of parameters, without sacrificing task accuracy. Importantly, this result is nearly identical to that of full-scope fine-tuning (for example, above 98 percent language consistency for both methods across all prompt scenarios) but uses a fraction of the computational resources. To the best of our knowledge, this is the first approach to leverage layer-localization of language control for efficient multilingual adaptation. 

---
# Rewarding Intellectual Humility Learning When Not To Answer In Large Language Models 

**Authors**: Abha Jha, Akanksha Mahajan, Ashwath Vaithinathan Aravindan, Praveen Saravanan, Sai Sailaja Policharla, Sonal Chaturbhuj Gehlot  

**Link**: [PDF](https://arxiv.org/pdf/2601.20126)  

**Abstract**: Large Language Models (LLMs) often produce hallucinated or unverifiable content, undermining their reliability in factual domains. This work investigates Reinforcement Learning with Verifiable Rewards (RLVR) as a training paradigm that explicitly rewards abstention ("I don't know") alongside correctness to promote intellectual humility. We fine-tune and evaluate Granite-3.3-2B-Instruct and Qwen-3-4B-Instruct on the MedMCQA and Hendrycks Math benchmarks using a ternary reward structure ($-1$, r_abs, 1) under varying abstention reward structures. We further study the effect of combining RLVR with supervised fine-tuning strategies that teach abstention prior to reinforcement learning. Our results show that moderate abstention rewards (r_abs $\approx -0.25$ to 0.3) consistently reduce incorrect responses without severe accuracy degradation on multiple-choice tasks, with larger models exhibiting greater robustness to abstention incentives. On open-ended question answering, we observe limitations due to insufficient exploration, which can be partially mitigated through supervised abstention training. Overall, these findings demonstrate the feasibility and flexibility of verifiable reward design as a practical approach for hallucination mitigation in language models. Reproducible code for our abstention training framework is available here this https URL. 

---
# Quantifying non deterministic drift in large language models 

**Authors**: Claire Nicholson  

**Link**: [PDF](https://arxiv.org/pdf/2601.19934)  

**Abstract**: Large language models (LLMs) are widely used for tasks ranging from summarisation to decision support. In practice, identical prompts do not always produce identical outputs, even when temperature and other decoding parameters are fixed. In this work, we conduct repeated-run experiments to empirically quantify baseline behavioural drift, defined as output variability observed when the same prompt is issued multiple times under operator-free conditions. We evaluate two publicly accessible models, gpt-4o-mini and llama3.1-8b, across five prompt categories using exact repeats, perturbed inputs, and reuse modes at temperatures of 0.0 and 0.7. Drift is measured using unique output fractions, lexical similarity, and word count statistics, enabling direct comparison across models, prompting modes, and deployment types. The results show that nondeterminism persists even at temperature 0.0, with distinct variability patterns by model size, deployment, and prompt type. We situate these findings within existing work on concept drift, behavioural drift, and infrastructure-induced nondeterminism, discuss the limitations of lexical metrics, and highlight emerging semantic approaches. By establishing a systematic empirical baseline in the absence of stabilisation techniques, this study provides a reference point for evaluating future drift mitigation and control methods. 

---
# Text-to-State Mapping for Non-Resolution Reasoning: The Contradiction-Preservation Principle 

**Authors**: Kei Saito  

**Link**: [PDF](https://arxiv.org/pdf/2601.19933)  

**Abstract**: Non-Resolution Reasoning (NRR) provides a formal framework for maintaining semantic ambiguity rather than forcing premature interpretation collapse. While the foundational architecture establishes state spaces and operators for ambiguity-preserving computation, the critical question of how natural language maps to these mathematical structures remains open. This paper introduces the text-to-state mapping function {\phi} that transforms linguistic input into superposition states within the NRR framework. We formalize the Contradiction-Preservation Principle, which requires that genuinely ambiguous expressions maintain non-zero entropy in their state representations, and develop extraction protocols using existing Large Language Models as interpretation generators. Empirical validation across 68 test sentences spanning lexical, structural, and pragmatic ambiguity demonstrates that our mapping achieves mean Shannon entropy H(S) = 1.087 bits for ambiguous inputs while baseline single-interpretation approaches yield H(S) = 0.000. The framework provides the missing algorithmic bridge between raw text and the formal state spaces on which NRR operators act, enabling architectural collapse deferment in language model inference. 

---
# Semantic Uncertainty Quantification of Hallucinations in LLMs: A Quantum Tensor Network Based Method 

**Authors**: Pragatheeswaran Vipulanandan, Kamal Premaratne, Dilip Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2601.20026)  

**Abstract**: Large language models (LLMs) exhibit strong generative capabilities but remain vulnerable to confabulations, fluent yet unreliable outputs that vary arbitrarily even under identical prompts. Leveraging a quantum tensor network based pipeline, we propose a quantum physics inspired uncertainty quantification framework that accounts for aleatoric uncertainty in token sequence probability for semantic equivalence based clustering of LLM generations. This offers a principled and interpretable scheme for hallucination detection. We further introduce an entropy maximization strategy that prioritizes high certainty, semantically coherent outputs and highlights entropy regions where LLM decisions are likely to be unreliable, offering practical guidelines for when human oversight is warranted. We evaluate the robustness of our scheme under different generation lengths and quantization levels, dimensions overlooked in prior studies, demonstrating that our approach remains reliable even in resource constrained deployments. A total of 116 experiments on TriviaQA, NQ, SVAMP, and SQuAD across multiple architectures including Mistral-7B, Mistral-7B-instruct, Falcon-rw-1b, LLaMA-3.2-1b, LLaMA-2-13b-chat, LLaMA-2-7b-chat, LLaMA-2-13b, and LLaMA-2-7b show consistent improvements in AUROC and AURAC over state of the art baselines. 

---
# Evaluating Large Language Models for Abstract Evaluation Tasks: An Empirical Study 

**Authors**: Yinuo Liu, Emre Sezgin, Eric A. Youngstrom  

**Link**: [PDF](https://arxiv.org/pdf/2601.19925)  

**Abstract**: Introduction: Large language models (LLMs) can process requests and generate texts, but their feasibility for assessing complex academic content needs further investigation. To explore LLM's potential in assisting scientific review, this study examined ChatGPT-5, Gemini-3-Pro, and Claude-Sonnet-4.5's consistency and reliability in evaluating abstracts compared to one another and to human reviewers. Methods: 160 abstracts from a local conference were graded by human reviewers and three LLMs using one rubric. Composite score distributions across three LLMs and fourteen reviewers were examined. Inter-rater reliability was calculated using intraclass correlation coefficients (ICCs) for within-AI reliability and AI-human concordance. Bland-Altman plots were examined for visual agreement patterns and systematic bias. Results: LLMs achieved good-to-excellent agreement with each other (ICCs: 0.59-0.87). ChatGPT and Claude reached moderate agreement with human reviewers on overall quality and content-specific criteria, with ICCs ~.45-.60 for composite, impression, clarity, objective, and results. They exhibited fair agreement on subjective dimensions, with ICC ranging from 0.23-0.38 for impact, engagement, and applicability. Gemini showed fair agreement on half criteria and no reliability on impact and applicability. Three LLMs showed acceptable or negligible mean difference (ChatGPT=0.24, Gemini=0.42, Claude=-0.02) from the human mean composite scores. Discussion: LLMs could process abstracts in batches with moderate agreement with human experts on overall quality and objective criteria. With appropriate process architecture, they can apply a rubric consistently across volumes of abstracts exceeding feasibility for a human rater. The weaker performance on subjective dimensions indicates that AI should serve a complementary role in evaluation, while human expertise remains essential. 

---
# FFE-Hallu:Hallucinations in Fixed Figurative Expressions:Benchmark of Idioms and Proverbs in the Persian Language 

**Authors**: Faezeh Hosseini, Mohammadali Yousefzadeh, Yadollah Yaghoobzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2601.20105)  

**Abstract**: Figurative language, particularly fixed figurative expressions (FFEs) such as idioms and proverbs, poses persistent challenges for large language models (LLMs). Unlike literal phrases, FFEs are culturally grounded, largely non-compositional, and conventionally fixed, making them especially vulnerable to figurative hallucination. We define figurative hallucination as the generation or endorsement of expressions that sound idiomatic and plausible but do not exist as authentic figurative expressions in the target language. We introduce FFEHallu, the first comprehensive benchmark for evaluating figurative hallucination in LLMs, with a focus on Persian, a linguistically rich yet underrepresented language. FFEHallu consists of 600 carefully curated instances spanning three complementary tasks: (i) FFE generation from meaning, (ii) detection of fabricated FFEs across four controlled construction categories, and (iii) FFE to FFE translation from English to Persian. Evaluating six state of the art multilingual LLMs, we find systematic weaknesses in figurative competence and cultural grounding. While models such as GPT4.1 demonstrate relatively strong performance in rejecting fabricated FFEs and retrieving authentic ones, most models struggle to reliably distinguish real expressions from high quality fabrications and frequently hallucinate during cross lingual translation. These findings reveal substantial gaps in current LLMs handling of figurative language and underscore the need for targeted benchmarks to assess and mitigate figurative hallucination. 

---
# Attribution Techniques for Mitigating Hallucinated Information in RAG Systems: A Survey 

**Authors**: Yuqing Zhao, Ziyao Liu, Yongsen Zheng, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2601.19927)  

**Abstract**: Large Language Models (LLMs)-based question answering (QA) systems play a critical role in modern AI, demonstrating strong performance across various tasks. However, LLM-generated responses often suffer from hallucinations, unfaithful statements lacking reliable references. Retrieval-Augmented Generation (RAG) frameworks enhance LLM responses by incorporating external references but also introduce new forms of hallucination due to complex interactions between the retriever and generator. To address these challenges, researchers have explored attribution-based techniques that ensure responses are verifiably supported by retrieved content. Despite progress, a unified pipeline for these techniques, along with a clear taxonomy and systematic comparison of their strengths and weaknesses, remains lacking. A well-defined taxonomy is essential for identifying specific failure modes within RAG systems, while comparative analysis helps practitioners choose appropriate solutions based on hallucination types and application context. This survey investigates how attribution-based techniques are used within RAG systems to mitigate hallucinations and addresses the gap by: (i) outlining a taxonomy of hallucination types in RAG systems, (ii) presenting a unified pipeline for attribution techniques, (iii) reviewing techniques based on the hallucinations they target, and (iv) discussing strengths and weaknesses with practical guidelines. This work offers insights for future research and practical use of attribution techniques in RAG systems. 

---
# Table-BiEval: A Self-Supervised, Dual-Track Framework for Decoupling Structure and Content in LLM Evaluation 

**Authors**: Boxiang Zhao, Qince Li, Zhonghao Wang, Zelin Cao, Yi Wang, Peng Cheng, Bo Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.19923)  

**Abstract**: As Large Language Models (LLMs) evolve into autonomous agents, the capability to faithfully translate natural language into rigorous structured formats-essential for tool invocation-and to convert complex tabular information into machine-readable specifications has become paramount. However, current evaluations lack effective methodologies to measure this structural fidelity without costly human intervention, as traditional text metrics fail to detect semantic drift in code-like outputs. This paper proposes Table-BiEval, a novel approach based on a human-free, self-supervised evaluation framework, to assess LLMs performance quantitatively. By leveraging deterministic Intermediate Representations, our framework calculates Content Semantic Accuracy and Normalized Tree Edit Distance to decouple structure from content. Also, it empirically evaluates 15 state-of-the-art LLMs across dual topological dimensions-hierarchical structures and flat tables. The results reveal substantial variability, highlighting that mid-sized models can surprisingly outperform larger counterparts in structural efficiency and confirming that deep recursive nesting remains a universal bottleneck for current architectures. 

---
# Demystifying Multi-Agent Debate: The Role of Confidence and Diversity 

**Authors**: Xiaochen Zhu, Caiqi Zhang, Yizhou Chi, Tom Stafford, Nigel Collier, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2601.19921)  

**Abstract**: Multi-agent debate (MAD) is widely used to improve large language model (LLM) performance through test-time scaling, yet recent work shows that vanilla MAD often underperforms simple majority vote despite higher computational cost. Studies show that, under homogeneous agents and uniform belief updates, debate preserves expected correctness and therefore cannot reliably improve outcomes. Drawing on findings from human deliberation and collective decision-making, we identify two key mechanisms missing from vanilla MAD: (i) diversity of initial viewpoints and (ii) explicit, calibrated confidence communication. We propose two lightweight interventions. First, a diversity-aware initialisation that selects a more diverse pool of candidate answers, increasing the likelihood that a correct hypothesis is present at the start of debate. Second, a confidence-modulated debate protocol in which agents express calibrated confidence and condition their updates on others' confidence. We show theoretically that diversity-aware initialisation improves the prior probability of MAD success without changing the underlying update dynamics, while confidence-modulated updates enable debate to systematically drift to the correct hypothesis. Empirically, across six reasoning-oriented QA benchmarks, our methods consistently outperform vanilla MAD and majority vote. Our results connect human deliberation with LLM-based debate and demonstrate that simple, principled modifications can substantially enhance debate effectiveness. 

---
# OPT-Engine: Benchmarking the Limits of LLMs in Optimization Modeling via Complexity Scaling 

**Authors**: Yitian Chen, Cheng Cheng, Yinan Sun, Zi Ling, Dongdong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2601.19924)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive progress in optimization modeling, fostering a rapid expansion of new methodologies and evaluation benchmarks. However, the boundaries of their capabilities in automated formulation and problem solving remain poorly understood, particularly when extending to complex, real-world tasks. To bridge this gap, we propose OPT-ENGINE, an extensible benchmark framework designed to evaluate LLMs on optimization modeling with controllable and scalable difficulty levels. OPT-ENGINE spans 10 canonical tasks across operations research, with five Linear Programming and five Mixed-Integer Programming. Utilizing OPT-ENGINE, we conduct an extensive study of LLMs' reasoning capabilities, addressing two critical questions: 1.) Do LLMs' performance remain robust when generalizing to out-of-distribution optimization tasks that scale in complexity beyond current benchmark levels? and 2.) At what stage, from problem interpretation to solution generation, do current LLMs encounter the most significant bottlenecks? Our empirical results yield two key insights: first, tool-integrated reasoning with external solvers exhibits significantly higher robustness as task complexity escalates, while pure-text reasoning reaches a ceiling; second, the automated formulation of constraints constitutes the primary performance bottleneck. These findings provide actionable guidance for developing next-generation LLMs for advanced optimization. Our code is publicly available at \textcolor{blue}{this https URL}. 

---
# HEART: A Unified Benchmark for Assessing Humans and LLMs in Emotional Support Dialogue 

**Authors**: Laya Iyer, Kriti Aggarwal, Sanmi Koyejo, Gail Heyman, Desmond C. Ong, Subhabrata Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2601.19922)  

**Abstract**: Supportive conversation depends on skills that go beyond language fluency, including reading emotions, adjusting tone, and navigating moments of resistance, frustration, or distress. Despite rapid progress in language models, we still lack a clear way to understand how their abilities in these interpersonal domains compare to those of humans. We introduce HEART, the first-ever framework that directly compares humans and LLMs on the same multi-turn emotional-support conversations. For each dialogue history, we pair human and model responses and evaluate them through blinded human raters and an ensemble of LLM-as-judge evaluators. All assessments follow a rubric grounded in interpersonal communication science across five dimensions: Human Alignment, Empathic Responsiveness, Attunement, Resonance, and Task-Following. HEART uncovers striking behavioral patterns. Several frontier models approach or surpass the average human responses in perceived empathy and consistency. At the same time, humans maintain advantages in adaptive reframing, tension-naming, and nuanced tone shifts, particularly in adversarial turns. Human and LLM-as-judge preferences align on about 80 percent of pairwise comparisons, matching inter-human agreement, and their written rationales emphasize similar HEART dimensions. This pattern suggests an emerging convergence in the criteria used to assess supportive quality. By placing humans and models on equal footing, HEART reframes supportive dialogue as a distinct capability axis, separable from general reasoning or linguistic fluency. It provides a unified empirical foundation for understanding where model-generated support aligns with human social judgment, where it diverges, and how affective conversational competence scales with model size. 

---
# Lowest Span Confidence: A Zero-Shot Metric for Efficient and Black-Box Hallucination Detection in LLMs 

**Authors**: Yitong Qiao, Licheng Pan, Yu Mi, Lei Liu, Yue Shen, Fei Sun, Zhixuan Chu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19918)  

**Abstract**: Hallucinations in Large Language Models (LLMs), i.e., the tendency to generate plausible but non-factual content, pose a significant challenge for their reliable deployment in high-stakes environments. However, existing hallucination detection methods generally operate under unrealistic assumptions, i.e., either requiring expensive intensive sampling strategies for consistency checks or white-box LLM states, which are unavailable or inefficient in common API-based scenarios. To this end, we propose a novel efficient zero-shot metric called Lowest Span Confidence (LSC) for hallucination detection under minimal resource assumptions, only requiring a single forward with output probabilities. Concretely, LSC evaluates the joint likelihood of semantically coherent spans via a sliding window mechanism. By identifying regions of lowest marginal confidence across variable-length n-grams, LSC could well capture local uncertainty patterns strongly correlated with factual inconsistency. Importantly, LSC can mitigate the dilution effect of perplexity and the noise sensitivity of minimum token probability, offering a more robust estimate of factual uncertainty. Extensive experiments across multiple state-of-the-art (SOTA) LLMs and diverse benchmarks show that LSC consistently outperforms existing zero-shot baselines, delivering strong detection performance even under resource-constrained conditions. 

---
# Training Reasoning Models on Saturated Problems via Failure-Prefix Conditioning 

**Authors**: Minwu Kim, Safal Shrestha, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2601.20829)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has substantially improved the reasoning abilities of large language models (LLMs), yet training often stalls as problems become saturated. We identify the core challenge as the poor accessibility of informative failures: learning signals exist but are rarely encountered during standard rollouts. To address this, we propose failure-prefix conditioning, a simple and effective method for learning from saturated problems. Rather than starting from the original question, our approach reallocates exploration by conditioning training on prefixes derived from rare incorrect reasoning trajectories, thereby exposing the model to failure-prone states. We observe that failure-prefix conditioning yields performance gains matching those of training on medium-difficulty problems, while preserving token efficiency. Furthermore, we analyze the model's robustness, finding that our method reduces performance degradation under misleading failure prefixes, albeit with a mild trade-off in adherence to correct early reasoning. Finally, we demonstrate that an iterative approach, which refreshes failure prefixes during training, unlocks additional gains after performance plateaus. Overall, our results suggest that failure-prefix conditioning offers an effective pathway to extend RLVR training on saturated problems. 

---
# Reward Models Inherit Value Biases from Pretraining 

**Authors**: Brian Christian, Jessica A. F. Thompson, Elle Michelle Yang, Vincent Adam, Hannah Rose Kirk, Christopher Summerfield, Tsvetomira Dumbalska  

**Link**: [PDF](https://arxiv.org/pdf/2601.20838)  

**Abstract**: Reward models (RMs) are central to aligning large language models (LLMs) with human values but have received less attention than pre-trained and post-trained LLMs themselves. Because RMs are initialized from LLMs, they inherit representations that shape their behavior, but the nature and extent of this influence remain understudied. In a comprehensive study of 10 leading open-weight RMs using validated psycholinguistic corpora, we show that RMs exhibit significant differences along multiple dimensions of human value as a function of their base model. Using the "Big Two" psychological axes, we show a robust preference of Llama RMs for "agency" and a corresponding robust preference of Gemma RMs for "communion." This phenomenon holds even when the preference data and finetuning process are identical, and we trace it back to the logits of the respective instruction-tuned and pre-trained models. These log-probability differences themselves can be formulated as an implicit RM; we derive usable implicit reward scores and show that they exhibit the very same agency/communion difference. We run experiments training RMs with ablations for preference data source and quantity, which demonstrate that this effect is not only repeatable but surprisingly durable. Despite RMs being designed to represent human preferences, our evidence shows that their outputs are influenced by the pretrained LLMs on which they are based. This work underscores the importance of safety and alignment efforts at the pretraining stage, and makes clear that open-source developers' choice of base model is as much a consideration of values as of performance. 

---
# PaperAudit-Bench: Benchmarking Error Detection in Research Papers for Critical Automated Peer Review 

**Authors**: Songjun Tu, Yiwen Ma, Jiahao Lin, Qichao Zhang, Xiangyuan Lan, Junfeng.Li, Nan Xu, Linjing Li, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.19916)  

**Abstract**: Large language models can generate fluent peer reviews, yet their assessments often lack sufficient critical rigor when substantive issues are subtle and distributed across a paper. In this paper, we introduce PaperAudit-Bench, which consists of two components: (1) PaperAudit-Dataset, an error dataset covering both errors identifiable within individual sections and those requiring cross-section reasoning, designed for controlled evaluation under long-context settings; and (2) PaperAudit-Review, an automated review framework that integrates structured error detection with evidence-aware review generation to support critical assessment. Experiments on PaperAudit-Bench reveal large variability in error detectability across models and detection depths, highlighting the difficulty of identifying such errors under long-context settings. Relative to representative automated reviewing baselines, incorporating explicit error detection into the review workflow produces systematically stricter and more discriminative evaluations, demonstrating its suitability for peer review. Finally, we show that the dataset supports training lightweight LLM detectors via SFT and RL, enabling effective error detection at reduced computational cost. 

---
# GDCNet: Generative Discrepancy Comparison Network for Multimodal Sarcasm Detection 

**Authors**: Shuguang Zhang, Junhong Lian, Guoxin Yu, Baoxun Xu, Xiang Ao  

**Link**: [PDF](https://arxiv.org/pdf/2601.20618)  

**Abstract**: Multimodal sarcasm detection (MSD) aims to identify sarcasm within image-text pairs by modeling semantic incongruities across modalities. Existing methods often exploit cross-modal embedding misalignment to detect inconsistency but struggle when visual and textual content are loosely related or semantically indirect. While recent approaches leverage large language models (LLMs) to generate sarcastic cues, the inherent diversity and subjectivity of these generations often introduce noise. To address these limitations, we propose the Generative Discrepancy Comparison Network (GDCNet). This framework captures cross-modal conflicts by utilizing descriptive, factually grounded image captions generated by Multimodal LLMs (MLLMs) as stable semantic anchors. Specifically, GDCNet computes semantic and sentiment discrepancies between the generated objective description and the original text, alongside measuring visual-textual fidelity. These discrepancy features are then fused with visual and textual representations via a gated module to adaptively balance modality contributions. Extensive experiments on MSD benchmarks demonstrate GDCNet's superior accuracy and robustness, establishing a new state-of-the-art on the MMSD2.0 benchmark. 

---
# PILOT: Planning via Internalized Latent Optimization Trajectories for Large Language Models 

**Authors**: Haoyu Zheng, Yun Zhu, Yuqian Yuan, Bo Yuan, Wenqiao Zhang, Siliang Tang, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2601.19917)  

**Abstract**: Strategic planning is critical for multi-step reasoning, yet compact Large Language Models (LLMs) often lack the capacity to formulate global strategies, leading to error propagation in long-horizon tasks. Our analysis reveals that LLMs possess latent reasoning capabilities that can be unlocked when conditioned on explicit plans from a teacher model; however, runtime reliance on external guidance is often impractical due to latency and availability constraints. To bridge this gap, we propose PILOT (Planning via Internalized Latent Optimization Trajectories), a non-invasive framework designed to internalize the strategic oversight of large models into intrinsic Latent Guidance. Instead of altering backbone weights, PILOT employs a lightweight Hyper-Network to synthesize a query-conditioned Latent Guidance vector. This vector acts as an internal steering mechanism, guiding the model's representations toward optimal reasoning paths. Extensive experiments on mathematical and coding benchmarks demonstrate that PILOT effectively stabilizes reasoning trajectories, consistently outperforming strong baselines (e.g., +8.9% on MATH500) with negligible inference latency. 

---
# Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction 

**Authors**: Tianyi Alex Qiu, Micah Carroll, Cameron Allen  

**Link**: [PDF](https://arxiv.org/pdf/2601.20299)  

**Abstract**: The evaluation and post-training of large language models (LLMs) rely on supervision, but strong supervision for difficult tasks is often unavailable, especially when evaluating frontier models. In such cases, models are demonstrated to exploit evaluations built on such imperfect supervision, leading to deceptive results. However, underutilized in LLM research, a wealth of mechanism design research focuses on game-theoretic incentive compatibility, i.e., eliciting honest and informative answers with weak supervision. Drawing from this literature, we introduce the peer prediction method for model evaluation and post-training. It rewards honest and informative answers over deceptive and uninformative ones, using a metric based on mutual predictability and without requiring ground truth labels. We demonstrate the method's effectiveness and resistance to deception, with both theoretical guarantees and empirical validation on models with up to 405B parameters. We show that training an 8B model with peer prediction-based reward recovers most of the drop in truthfulness due to prior malicious finetuning, even when the reward is produced by a 0.135B language model with no finetuning. On the evaluation front, in contrast to LLM-as-a-Judge which requires strong and trusted judges, we discover an inverse scaling property in peer prediction, where, surprisingly, resistance to deception is strengthened as the capability gap between the experts and participants widens, enabling reliable evaluation of strong models with weak supervision. In particular, LLM-as-a-Judge become worse than random guess when facing deceptive models 5-20x the judge's size, while peer prediction thrives when such gaps are large, including in cases with over 100x size difference. 

---
# Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation 

**Authors**: Yanqi Dai, Yuxiang Ji, Xiao Zhang, Yong Wang, Xiangxiang Chu, Zhiwu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.20614)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) offers a robust mechanism for enhancing mathematical reasoning in large models. However, we identify a systematic lack of emphasis on more challenging questions in existing methods from both algorithmic and data perspectives, despite their importance for refining underdeveloped capabilities. Algorithmically, widely used Group Relative Policy Optimization (GRPO) suffers from an implicit imbalance where the magnitude of policy updates is lower for harder questions. Data-wise, augmentation approaches primarily rephrase questions to enhance diversity without systematically increasing intrinsic difficulty. To address these issues, we propose a two-dual MathForge framework to improve mathematical reasoning by targeting harder questions from both perspectives, which comprises a Difficulty-Aware Group Policy Optimization (DGPO) algorithm and a Multi-Aspect Question Reformulation (MQR) strategy. Specifically, DGPO first rectifies the implicit imbalance in GRPO via difficulty-balanced group advantage estimation, and further prioritizes harder questions by difficulty-aware question-level weighting. Meanwhile, MQR reformulates questions across multiple aspects to increase difficulty while maintaining the original gold answer. Overall, MathForge forms a synergistic loop: MQR expands the data frontier, and DGPO effectively learns from the augmented data. Extensive experiments show that MathForge significantly outperforms existing methods on various mathematical reasoning tasks. The code and augmented data are all available at this https URL. 

---
# LLM-AutoDP: Automatic Data Processing via LLM Agents for Model Fine-tuning 

**Authors**: Wei Huang, Anda Cheng, Yinggui Wang, Lei Wang, Tao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.20375)  

**Abstract**: Large Language Models (LLMs) can be fine-tuned on domain-specific data to enhance their performance in specialized fields. However, such data often contains numerous low-quality samples, necessitating effective data processing (DP). In practice, DP strategies are typically developed through iterative manual analysis and trial-and-error adjustment. These processes inevitably incur high labor costs and may lead to privacy issues in high-privacy domains like healthcare due to direct human access to sensitive data. Thus, achieving automated data processing without exposing the raw data has become a critical challenge. To address this challenge, we propose LLM-AutoDP, a novel framework that leverages LLMs as agents to automatically generate and optimize data processing strategies. Our method generates multiple candidate strategies and iteratively refines them using feedback signals and comparative evaluations. This iterative in-context learning mechanism enables the agent to converge toward high-quality processing pipelines without requiring direct human intervention or access to the underlying data. To further accelerate strategy search, we introduce three key techniques: Distribution Preserving Sampling, which reduces data volume while maintaining distributional integrity; Processing Target Selection, which uses a binary classifier to identify low-quality samples for focused processing; Cache-and-Reuse Mechanism}, which minimizes redundant computations by reusing prior processing results. Results show that models trained on data processed by our framework achieve over 80% win rates against models trained on unprocessed data. Compared to AutoML baselines based on LLM agents, LLM-AutoDP achieves approximately a 65% win rate. Moreover, our acceleration techniques reduce the total searching time by up to 10 times, demonstrating both effectiveness and efficiency. 

---
# PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs 

**Authors**: Oguzhan Gungordu, Siheng Xiong, Faramarz Fekri  

**Link**: [PDF](https://arxiv.org/pdf/2601.20539)  

**Abstract**: Large Language Models (LLMs) have enabled automated heuristic design (AHD) for combinatorial optimization problems (COPs), but existing frameworks' reliance on fixed evolutionary rules and static prompt templates often leads to myopic heuristic generation, redundant evaluations, and limited reasoning about how new heuristics should be derived. We propose a novel multi-agent reasoning framework, referred to as Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs (PathWise), which formulates heuristic generation as a sequential decision process over an entailment graph serving as a compact, stateful memory of the search trajectory. This approach allows the system to carry forward past decisions and reuse or avoid derivation information across generations. A policy agent plans evolutionary actions, a world model agent generates heuristic rollouts conditioned on those actions, and critic agents provide routed reflections summarizing lessons from prior steps, shifting LLM-based AHD from trial-and-error evolution toward state-aware planning through reasoning. Experiments across diverse COPs show that PathWise converges faster to better heuristics, generalizes across different LLM backbones, and scales to larger problem sizes. 

---
# What's the plan? Metrics for implicit planning in LLMs and their application to rhyme generation and question answering 

**Authors**: Jim Maar, Denis Paperno, Callum Stuart McDougall, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2601.20164)  

**Abstract**: Prior work suggests that language models, while trained on next token prediction, show implicit planning behavior: they may select the next token in preparation to a predicted future token, such as a likely rhyming word, as supported by a prior qualitative study of Claude 3.5 Haiku using a cross-layer transcoder. We propose much simpler techniques for assessing implicit planning in language models. With case studies on rhyme poetry generation and question answering, we demonstrate that our methodology easily scales to many models. Across models, we find that the generated rhyme (e.g. "-ight") or answer to a question ("whale") can be manipulated by steering at the end of the preceding line with a vector, affecting the generation of intermediate tokens leading up to the rhyme or answer word. We show that implicit planning is a universal mechanism, present in smaller models than previously thought, starting from 1B parameters. Our methodology offers a widely applicable direct way to study implicit planning abilities of LLMs. More broadly, understanding planning abilities of language models can inform decisions in AI safety and control. 

---
# CtrlCoT: Dual-Granularity Chain-of-Thought Compression for Controllable Reasoning 

**Authors**: Zhenxuan Fan, Jie Cao, Yang Dai, Zheqi Lv, Wenqiao Zhang, Zhongle Xie, Peng LU, Beng Chin Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2601.20467)  

**Abstract**: Chain-of-thought (CoT) prompting improves LLM reasoning but incurs high latency and memory cost due to verbose traces, motivating CoT compression with preserved correctness. Existing methods either shorten CoTs at the semantic level, which is often conservative, or prune tokens aggressively, which can miss task-critical cues and degrade accuracy. Moreover, combining the two is non-trivial due to sequential dependency, task-agnostic pruning, and distribution mismatch. We propose \textbf{CtrlCoT}, a dual-granularity CoT compression framework that harmonizes semantic abstraction and token-level pruning through three components: Hierarchical Reasoning Abstraction produces CoTs at multiple semantic granularities; Logic-Preserving Distillation trains a logic-aware pruner to retain indispensable reasoning cues (e.g., numbers and operators) across pruning ratios; and Distribution-Alignment Generation aligns compressed traces with fluent inference-time reasoning styles to avoid fragmentation. On MATH-500 with Qwen2.5-7B-Instruct, CtrlCoT uses 30.7\% fewer tokens while achieving 7.6 percentage points higher than the strongest baseline, demonstrating more efficient and reliable reasoning. Our code will be publicly available at this https URL. 

---
# Insight Agents: An LLM-Based Multi-Agent System for Data Insights 

**Authors**: Jincheng Bai, Zhenyu Zhang, Jennifer Zhang, Zhihuai Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.20048)  

**Abstract**: Today, E-commerce sellers face several key challenges, including difficulties in discovering and effectively utilizing available programs and tools, and struggling to understand and utilize rich data from various tools. We therefore aim to develop Insight Agents (IA), a conversational multi-agent Data Insight system, to provide E-commerce sellers with personalized data and business insights through automated information retrieval. Our hypothesis is that IA will serve as a force multiplier for sellers, thereby driving incremental seller adoption by reducing the effort required and increase speed at which sellers make good business decisions. In this paper, we introduce this novel LLM-backed end-to-end agentic system built on a plan-and-execute paradigm and designed for comprehensive coverage, high accuracy, and low latency. It features a hierarchical multi-agent structure, consisting of manager agent and two worker agents: data presentation and insight generation, for efficient information retrieval and problem-solving. We design a simple yet effective ML solution for manager agent that combines Out-of-Domain (OOD) detection using a lightweight encoder-decoder model and agent routing through a BERT-based classifier, optimizing both accuracy and latency. Within the two worker agents, a strategic planning is designed for API-based data model that breaks down queries into granular components to generate more accurate responses, and domain knowledge is dynamically injected to to enhance the insight generator. IA has been launched for Amazon sellers in US, which has achieved high accuracy of 90% based on human evaluation, with latency of P90 below 15s. 

---
# SokoBench: Evaluating Long-Horizon Planning and Reasoning in Large Language Models 

**Authors**: Sebastiano Monti, Carlo Nicolini, Gianni Pellegrini, Jacopo Staiano, Bruno Lepri  

**Link**: [PDF](https://arxiv.org/pdf/2601.20856)  

**Abstract**: Although the capabilities of large language models have been increasingly tested on complex reasoning tasks, their long-horizon planning abilities have not yet been extensively investigated. In this work, we provide a systematic assessment of the planning and long-horizon reasoning capabilities of state-of-the-art Large Reasoning Models (LRMs). We propose a novel benchmark based on Sokoban puzzles, intentionally simplified to isolate long-horizon planning from state persistence. Our findings reveal a consistent degradation in planning performance when more than 25 moves are required to reach the solution, suggesting a fundamental constraint on forward planning capacity. We show that equipping LRMs with Planning Domain Definition Language (PDDL) parsing, validation, and solving tools allows for modest improvements, suggesting inherent architectural limitations which might not be overcome by test-time scaling approaches alone. 

---
# MemCtrl: Using MLLMs as Active Memory Controllers on Embodied Agents 

**Authors**: Vishnu Sashank Dorbala, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2601.20831)  

**Abstract**: Foundation models rely on in-context learning for personalized decision making. The limited size of this context window necessitates memory compression and retrieval systems like RAG. These systems however often treat memory as large offline storage spaces, which is unfavorable for embodied agents that are expected to operate under strict memory and compute constraints, online. In this work, we propose MemCtrl, a novel framework that uses Multimodal Large Language Models (MLLMs) for pruning memory online. MemCtrl augments MLLMs with a trainable memory head \mu that acts as a gate to determine which observations or reflections to retain, update, or discard during exploration. We evaluate with training two types of \mu, 1) via an offline expert, and 2) via online RL, and observe significant improvement in overall embodied task completion ability on \mu-augmented MLLMs. In particular, on augmenting two low performing MLLMs with MemCtrl on multiple subsets of the EmbodiedBench benchmark, we observe that \mu-augmented MLLMs show an improvement of around 16% on average, with over 20% on specific instruction subsets. Finally, we present a qualitative analysis on the memory fragments collected by \mu, noting the superior performance of \mu augmented MLLMs on long and complex instruction types. 

---
# Dialogical Reasoning Across AI Architectures: A Multi-Model Framework for Testing AI Alignment Strategies 

**Authors**: Gray Cox  

**Link**: [PDF](https://arxiv.org/pdf/2601.20604)  

**Abstract**: This paper introduces a methodological framework for empirically testing AI alignment strategies through structured multi-model dialogue. Drawing on Peace Studies traditions - particularly interest-based negotiation, conflict transformation, and commons governance - we operationalize Viral Collaborative Wisdom (VCW), an approach that reframes alignment from a control problem to a relationship problem developed through dialogical reasoning.
Our experimental design assigns four distinct roles (Proposer, Responder, Monitor, Translator) to different AI systems across six conditions, testing whether current large language models can engage substantively with complex alignment frameworks. Using Claude, Gemini, and GPT-4o, we conducted 72 dialogue turns totaling 576,822 characters of structured exchange.
Results demonstrate that AI systems can engage meaningfully with Peace Studies concepts, surface complementary objections from different architectural perspectives, and generate emergent insights not present in initial framings - including the novel synthesis of "VCW as transitional framework." Cross-architecture patterns reveal that different models foreground different concerns: Claude emphasized verification challenges, Gemini focused on bias and scalability, and GPT-4o highlighted implementation barriers.
The framework provides researchers with replicable methods for stress-testing alignment proposals before implementation, while the findings offer preliminary evidence about AI capacity for the kind of dialogical reasoning VCW proposes. We discuss limitations, including the observation that dialogues engaged more with process elements than with foundational claims about AI nature, and outline directions for future research including human-AI hybrid protocols and extended dialogue studies. 

---
# Policy of Thoughts: Scaling LLM Reasoning via Test-time Policy Evolution 

**Authors**: Zhengbo Jiao, Hongyu Xian, Qinglong Wang, Yunpu Ma, Zhebo Wang, Zifan Zhang, Dezhang Kong, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.20379)  

**Abstract**: Large language models (LLMs) struggle with complex, long-horizon reasoning due to instability caused by their frozen policy assumption. Current test-time scaling methods treat execution feedback merely as an external signal for filtering or rewriting trajectories, without internalizing it to improve the underlying reasoning strategy. Inspired by Popper's epistemology of "conjectures and refutations," we argue that intelligence requires real-time evolution of the model's policy through learning from failed attempts. We introduce Policy of Thoughts (PoT), a framework that recasts reasoning as a within-instance online optimization process. PoT first generates diverse candidate solutions via an efficient exploration mechanism, then uses Group Relative Policy Optimization (GRPO) to update a transient LoRA adapter based on execution feedback. This closed-loop design enables dynamic, instance-specific refinement of the model's reasoning priors. Experiments show that PoT dramatically boosts performance: a 4B model achieves 49.71% accuracy on LiveCodeBench, outperforming GPT-4o and DeepSeek-V3 despite being over 50 smaller. 

---
# Should I Have Expressed a Different Intent? Counterfactual Generation for LLM-Based Autonomous Control 

**Authors**: Amirmohammad Farzaneh, Salvatore D'Oro, Osvaldo Simeone  

**Link**: [PDF](https://arxiv.org/pdf/2601.20090)  

**Abstract**: Large language model (LLM)-powered agents can translate high-level user intents into plans and actions in an environment. Yet after observing an outcome, users may wonder: What if I had phrased my intent differently? We introduce a framework that enables such counterfactual reasoning in agentic LLM-driven control scenarios, while providing formal reliability guarantees. Our approach models the closed-loop interaction between a user, an LLM-based agent, and an environment as a structural causal model (SCM), and leverages test-time scaling to generate multiple candidate counterfactual outcomes via probabilistic abduction. Through an offline calibration phase, the proposed conformal counterfactual generation (CCG) yields sets of counterfactual outcomes that are guaranteed to contain the true counterfactual outcome with high probability. We showcase the performance of CCG on a wireless network control use case, demonstrating significant advantages compared to naive re-execution baselines. 

---
# Teaching LLMs to Ask: Self-Querying Category-Theoretic Planning for Under-Specified Reasoning 

**Authors**: Shuhui Qu  

**Link**: [PDF](https://arxiv.org/pdf/2601.20014)  

**Abstract**: Inference-time planning with large language models frequently breaks under partial observability: when task-critical preconditions are not specified at query time, models tend to hallucinate missing facts or produce plans that violate hard constraints. We introduce \textbf{Self-Querying Bidirectional Categorical Planning (SQ-BCP)}, which explicitly represents precondition status (\texttt{Sat}/\texttt{Viol}/\texttt{Unk}) and resolves unknowns via (i) targeted self-queries to an oracle/user or (ii) \emph{bridging} hypotheses that establish the missing condition through an additional action. SQ-BCP performs bidirectional search and invokes a pullback-based verifier as a categorical certificate of goal compatibility, while using distance-based scores only for ranking and pruning. We prove that when the verifier succeeds and hard constraints pass deterministic checks, accepted plans are compatible with goal requirements; under bounded branching and finite resolution depth, SQ-BCP finds an accepting plan when one exists. Across WikiHow and RecipeNLG tasks with withheld preconditions, SQ-BCP reduces resource-violation rates to \textbf{14.9\%} and \textbf{5.8\%} (vs.\ \textbf{26.0\%} and \textbf{15.7\%} for the best baseline), while maintaining competitive reference quality. 

---
# AMA: Adaptive Memory via Multi-Agent Collaboration 

**Authors**: Weiquan Huang, Zixuan Wang, Hehai Lin, Sudong Wang, Bo Xu, Qian Li, Beier Zhu, Linyi Yang, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.20352)  

**Abstract**: The rapid evolution of Large Language Model (LLM) agents has necessitated robust memory systems to support cohesive long-term interaction and complex reasoning. Benefiting from the strong capabilities of LLMs, recent research focus has shifted from simple context extension to the development of dedicated agentic memory systems. However, existing approaches typically rely on rigid retrieval granularity, accumulation-heavy maintenance strategies, and coarse-grained update mechanisms. These design choices create a persistent mismatch between stored information and task-specific reasoning demands, while leading to the unchecked accumulation of logical inconsistencies over time. To address these challenges, we propose Adaptive Memory via Multi-Agent Collaboration (AMA), a novel framework that leverages coordinated agents to manage memory across multiple granularities. AMA employs a hierarchical memory design that dynamically aligns retrieval granularity with task complexity. Specifically, the Constructor and Retriever jointly enable multi-granularity memory construction and adaptive query routing. The Judge verifies the relevance and consistency of retrieved content, triggering iterative retrieval when evidence is insufficient or invoking the Refresher upon detecting logical conflicts. The Refresher then enforces memory consistency by performing targeted updates or removing outdated entries. Extensive experiments on challenging long-context benchmarks show that AMA significantly outperforms state-of-the-art baselines while reducing token consumption by approximately 80% compared to full-context methods, demonstrating its effectiveness in maintaining retrieval precision and long-term memory consistency. 

---
# Fuzzy Categorical Planning: Autonomous Goal Satisfaction with Graded Semantic Constraints 

**Authors**: Shuhui Qu  

**Link**: [PDF](https://arxiv.org/pdf/2601.20021)  

**Abstract**: Natural-language planning often involves vague predicates (e.g., suitable substitute, stable enough) whose satisfaction is inherently graded. Existing category-theoretic planners provide compositional structure and pullback-based hard-constraint verification, but treat applicability as crisp, forcing thresholding that collapses meaningful distinctions and cannot track quality degradation across multi-step plans. We propose Fuzzy Category-theoretic Planning (FCP), which annotates each action (morphism) with a degree in [0,1], composes plan quality via a t-norm Lukasiewicz, and retains crisp executability checks via pullback verification. FCP grounds graded applicability from language using an LLM with k-sample median aggregation and supports meeting-in-the-middle search using residuum-based backward requirements. We evaluate on (i) public PDDL3 preference/oversubscription benchmarks and (ii) RecipeNLG-Subs, a missing-substitute recipe-planning benchmark built from RecipeNLG with substitution candidates from Recipe1MSubs and FoodKG. FCP improves success and reduces hard-constraint violations on RecipeNLG-Subs compared to LLM-only and ReAct-style baselines, while remaining competitive with classical PDDL3 planners. 

---
# HESTIA: A Hessian-Guided Differentiable Quantization-Aware Training Framework for Extremely Low-Bit LLMs 

**Authors**: Guoan Wang, Feiyu Wang, Zongwei Lv, Yikun Zong, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.20745)  

**Abstract**: As large language models (LLMs) continue to scale, deployment is increasingly bottlenecked by the memory wall, motivating a shift toward extremely low-bit quantization. However, most quantization-aware training (QAT) methods apply hard rounding and the straight-through estimator (STE) from the beginning of the training, which prematurely discretizes the optimization landscape and induces persistent gradient mismatch between latent weights and quantized weights, hindering effective optimization of quantized models. To address this, we propose Hestia, a Hessian-guided differentiable QAT framework for extremely low-bit LLMs, which replaces the rigid step function with a temperature-controlled softmax relaxation to maintain gradient flow early in training while progressively hardening quantization. Furthermore, Hestia leverages a tensor-wise Hessian trace metric as a lightweight curvature signal to drive fine-grained temperature annealing, enabling sensitivity-aware discretization across the model. Evaluations on Llama-3.2 show that Hestia consistently outperforms existing ternary QAT baselines, yielding average zero-shot improvements of 5.39% and 4.34% for the 1B and 3B models. These results indicate that Hessian-guided relaxation effectively recovers representational capacity, establishing a more robust training path for 1.58-bit LLMs. The code is available at this https URL. 

---
# LEMON: How Well Do MLLMs Perform Temporal Multimodal Understanding on Instructional Videos? 

**Authors**: Zhuang Yu, Lei Shen, Jing Zhao, Shiliang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.20705)  

**Abstract**: Recent multimodal large language models (MLLMs) have shown remarkable progress across vision, audio, and language tasks, yet their performance on long-form, knowledge-intensive, and temporally structured educational content remains largely unexplored. To bridge this gap, we introduce LEMON, a Lecture-based Evaluation benchmark for MultimOdal uNderstanding, focusing on STEM lecture videos that require long-horizon reasoning and cross-modal integration. LEMON comprises 2,277 video segments spanning 5 disciplines and 29 courses, with an average duration of 196.1 seconds, yielding 4,181 high-quality QA pairs, including 3,413 multiple-choice and 768 open-ended questions. Distinct from existing video benchmarks, LEMON features: (1) semantic richness and disciplinary density, (2) tightly coupled video-audio-text modalities, (3) explicit temporal and pedagogical structure, and (4) contextually linked multi-turn questioning. It further encompasses six major tasks and twelve subtasks, covering the full cognitive spectrum from perception to reasoning and then to generation. Comprehensive experiments reveal substantial performance gaps across tasks, highlighting that even state-of-the-art MLLMs like GPT-4o struggle with temporal reasoning and instructional prediction. We expect LEMON to serve as an extensible and challenging benchmark for advancing multimodal perception, reasoning, and generation in long-form instructional contents. 

---
# Beyond GEMM-Centric NPUs: Enabling Efficient Diffusion LLM Sampling 

**Authors**: Binglei Lou, Haoran Wu, Yao Lai, Jiayi Nie, Can Xiao, Xuan Guo, Rika Antonova, Robert Mullins, Aaron Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.20706)  

**Abstract**: Diffusion Large Language Models (dLLMs) introduce iterative denoising to enable parallel token generation, but their sampling phase displays fundamentally different characteristics compared to GEMM-centric transformer layers. Profiling on modern GPUs reveals that sampling can account for up to 70% of total model inference latency-primarily due to substantial memory loads and writes from vocabulary-wide logits, reduction-based token selection, and iterative masked updates. These processes demand large on-chip SRAM and involve irregular memory accesses that conventional NPUs struggle to handle efficiently. To address this, we identify a set of critical instructions that an NPU architecture must specifically optimize for dLLM sampling. Our design employs lightweight non-GEMM vector primitives, in-place memory reuse strategies, and a decoupled mixed-precision memory hierarchy. Together, these optimizations deliver up to a 2.53x speedup over the NVIDIA RTX A6000 GPU under an equivalent nm technology node. We also open-source our cycle-accurate simulation and post-synthesis RTL verification code, confirming functional equivalence with current dLLM PyTorch implementations. 

---
# Audio Deepfake Detection in the Age of Advanced Text-to-Speech models 

**Authors**: Robin Singh, Aditya Yogesh Nair, Fabio Palumbo, Florian Barbaro, Anna Dyka, Lohith Rachakonda  

**Link**: [PDF](https://arxiv.org/pdf/2601.20510)  

**Abstract**: Recent advances in Text-to-Speech (TTS) systems have substantially increased the realism of synthetic speech, raising new challenges for audio deepfake detection. This work presents a comparative evaluation of three state-of-the-art TTS models--Dia2, Maya1, and MeloTTS--representing streaming, LLM-based, and non-autoregressive architectures. A corpus of 12,000 synthetic audio samples was generated using the Daily-Dialog dataset and evaluated against four detection frameworks, including semantic, structural, and signal-level approaches. The results reveal significant variability in detector performance across generative mechanisms: models effective against one TTS architecture may fail against others, particularly LLM-based synthesis. In contrast, a multi-view detection approach combining complementary analysis levels demonstrates robust performance across all evaluated models. These findings highlight the limitations of single-paradigm detectors and emphasize the necessity of integrated detection strategies to address the evolving landscape of audio deepfake threats. 

---
# GuideAI: A Real-time Personalized Learning Solution with Adaptive Interventions 

**Authors**: Ananya Shukla, Chaitanya Modi, Satvik Bajpai, Siddharth Siddharth  

**Link**: [PDF](https://arxiv.org/pdf/2601.20402)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful learning tools, but they lack awareness of learners' cognitive and physiological states, limiting their adaptability to the user's learning style. Contemporary learning techniques primarily focus on structured learning paths, knowledge tracing, and generic adaptive testing but fail to address real-time learning challenges driven by cognitive load, attention fluctuations, and engagement levels. Building on findings from a formative user study (N=66), we introduce GuideAI, a multi-modal framework that enhances LLM-driven learning by integrating real-time biosensory feedback including eye gaze tracking, heart rate variability, posture detection, and digital note-taking behavior. GuideAI dynamically adapts learning content and pacing through cognitive optimizations (adjusting complexity based on learning progress markers), physiological interventions (breathing guidance and posture correction), and attention-aware strategies (redirecting focus using gaze analysis). Additionally, GuideAI supports diverse learning modalities, including text-based, image-based, audio-based, and video-based instruction, across varied knowledge domains. A preliminary study (N = 25) assessed GuideAI's impact on knowledge retention and cognitive load through standardized assessments. The results show statistically significant improvements in both problem-solving capability and recall-based knowledge assessments. Participants also experienced notable reductions in key NASA-TLX measures including mental demand, frustration levels, and effort, while simultaneously reporting enhanced perceived performance. These findings demonstrate GuideAI's potential to bridge the gap between current LLM-based learning systems and individualized learner needs, paving the way for adaptive, cognition-aware education at scale. 

---
# DiagLink: A Dual-User Diagnostic Assistance System by Synergizing Experts with LLMs and Knowledge Graphs 

**Authors**: Zihan Zhou, Yinan Liu, Yuyang Xie, Bin Wang, Xiaochun Yang, Zezheng Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.20311)  

**Abstract**: The global shortage and uneven distribution of medical expertise continue to hinder equitable access to accurate diagnostic care. While existing intelligent diagnostic system have shown promise, most struggle with dual-user interaction, and dynamic knowledge integration -- limiting their real-world applicability. In this study, we present DiagLink, a dual-user diagnostic assistance system that synergizes large language models (LLMs), knowledge graphs (KGs), and medical experts to support both patients and physicians. DiagLink uses guided dialogues to elicit patient histories, leverages LLMs and KGs for collaborative reasoning, and incorporates physician oversight for continuous knowledge validation and evolution. The system provides a role-adaptive interface, dynamically visualized history, and unified multi-source evidence to improve both trust and usability. We evaluate DiagLink through user study, use cases and expert interviews, demonstrating its effectiveness in improving user satisfaction and diagnostic efficiency, while offering insights for the design of future AI-assisted diagnostic systems. 

---
# Demonstration-Free Robotic Control via LLM Agents 

**Authors**: Brian Y. Tsui, Alan Y. Fang, Tiffany J. Hwu  

**Link**: [PDF](https://arxiv.org/pdf/2601.20334)  

**Abstract**: Robotic manipulation has increasingly adopted vision-language-action (VLA) models, which achieve strong performance but typically require task-specific demonstrations and fine-tuning, and often generalize poorly under domain shift. We investigate whether general-purpose large language model (LLM) agent frameworks, originally developed for software engineering, can serve as an alternative control paradigm for embodied manipulation. We introduce FAEA (Frontier Agent as Embodied Agent), which applies an LLM agent framework directly to embodied manipulation without modification. Using the same iterative reasoning that enables software agents to debug code, FAEA enables embodied agents to reason through manipulation strategies. We evaluate an unmodified frontier agent, Claude Agent SDK, across the LIBERO, ManiSkill3, and MetaWorld benchmarks. With privileged environment state access, FAEA achieves success rates of 84.9%, 85.7%, and 96%, respectively. This level of task success approaches that of VLA models trained with less than 100 demonstrations per task, without requiring demonstrations or fine-tuning. With one round of human feedback as an optional optimization, performance increases to 88.2% on LIBERO. This demonstration-free capability has immediate practical value: FAEA can autonomously explore novel scenarios in simulation and generate successful trajectories for training data augmentation in embodied learning. Our results indicate that general-purpose agents are sufficient for a class of manipulation tasks dominated by deliberative, task-level planning. This opens a path for robotics systems to leverage actively maintained agent infrastructure and benefit directly from ongoing advances in frontier models. Code is available at this https URL 

---
# Eliciting Least-to-Most Reasoning for Phishing URL Detection 

**Authors**: Holly Trikilis, Pasindu Marasinghe, Fariza Rashid, Suranga Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2601.20270)  

**Abstract**: Phishing continues to be one of the most prevalent attack vectors, making accurate classification of phishing URLs essential. Recently, large language models (LLMs) have demonstrated promising results in phishing URL detection. However, their reasoning capabilities that enabled such performance remain underexplored. To this end, in this paper, we propose a Least-to-Most prompting framework for phishing URL detection. In particular, we introduce an "answer sensitivity" mechanism that guides Least-to-Most's iterative approach to enhance reasoning and yield higher prediction accuracy. We evaluate our framework using three URL datasets and four state-of-the-art LLMs, comparing against a one-shot approach and a supervised model. We demonstrate that our framework outperforms the one-shot baseline while achieving performance comparable to that of the supervised model, despite requiring significantly less training data. Furthermore, our in-depth analysis highlights how the iterative reasoning enabled by Least-to-Most, and reinforced by our answer sensitivity mechanism, drives these performance gains. Overall, we show that this simple yet powerful prompting strategy consistently outperforms both one-shot and supervised approaches, despite requiring minimal training or few-shot guidance. Our experimental setup can be found in our Github repository this http URL. 

---
# Large language models accurately predict public perceptions of support for climate action worldwide 

**Authors**: Nattavudh Powdthavee, Sandra J. Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2601.20141)  

**Abstract**: Although most people support climate action, widespread underestimation of others' support stalls individual and systemic changes. In this preregistered experiment, we test whether large language models (LLMs) can reliably predict these perception gaps worldwide. Using country-level indicators and public opinion data from 125 countries, we benchmark four state-of-the-art LLMs against Gallup World Poll 2021/22 data and statistical regressions. LLMs, particularly Claude, accurately capture public perceptions of others' willingness to contribute financially to climate action (MAE approximately 5 p.p.; r = .77), comparable to statistical models, though performance declines in less digitally connected, lower-GDP countries. Controlled tests show that LLMs capture the key psychological process - social projection with a systematic downward bias - and rely on structured reasoning rather than memorized values. Overall, LLMs provide a rapid tool for assessing perception gaps in climate action, serving as an alternative to costly surveys in resource-rich countries and as a complement in underrepresented populations. 

---
# Dynamics of Human-AI Collective Knowledge on the Web: A Scalable Model and Insights for Sustainable Growth 

**Authors**: Buddhika Nettasinghe, Kang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.20099)  

**Abstract**: Humans and large language models (LLMs) now co-produce and co-consume the web's shared knowledge archives. Such human-AI collective knowledge ecosystems contain feedback loops with both benefits (e.g., faster growth, easier learning) and systemic risks (e.g., quality dilution, skill reduction, model collapse). To understand such phenomena, we propose a minimal, interpretable dynamical model of the co-evolution of archive size, archive quality, model (LLM) skill, aggregate human skill, and query volume. The model captures two content inflows (human, LLM) controlled by a gate on LLM-content admissions, two learning pathways for humans (archive study vs. LLM assistance), and two LLM-training modalities (corpus-driven scaling vs. learning from human feedback). Through numerical experiments, we identify different growth regimes (e.g., healthy growth, inverted flow, inverted learning, oscillations), and show how platform and policy levers (gate strictness, LLM training, human learning pathways) shift the system across regime boundaries. Two domain configurations (PubMed, GitHub and Copilot) illustrate contrasting steady states under different growth rates and moderation norms. We also fit the model to Wikipedia's knowledge flow during pre-ChatGPT and post-ChatGPT eras separately. We find a rise in LLM additions with a concurrent decline in human inflow, consistent with a regime identified by the model. Our model and analysis yield actionable insights for sustainable growth of human-AI collective knowledge on the Web. 

---
# Bench4HLS: End-to-End Evaluation of LLMs in High-Level Synthesis Code Generation 

**Authors**: M Zafir Sadik Khan, Kimia Azar, Hadi Kamali  

**Link**: [PDF](https://arxiv.org/pdf/2601.19941)  

**Abstract**: In last two years, large language models (LLMs) have shown strong capabilities in code generation, including hardware design at register-transfer level (RTL). While their use in high-level synthesis (HLS) remains comparatively less mature, the ratio of HLS- to RTL-focused studies has shifted from 1:10 to 2:10 in the past six months, indicating growing interest in leveraging LLMs for high-level design entry while relying on downstream synthesis for optimization. This growing trend highlights the need for a comprehensive benchmarking and evaluation framework dedicated to LLM-based HLS. To address this, We present Bench4HLS for evaluating LLM-generated HLS designs. Bench4HLS comprises 170 manually drafted and validated case studies, spanning small kernels to complex accelerators, curated from widely used public repositories. The framework supports fully automated assessment of compilation success, functional correctness via simulation, and synthesis feasibility/optimization. Crucially, Bench4HLS integrates a pluggable API for power, performance, and area (PPA) analysis across various HLS toolchains and architectures, demonstrated here with Xilinx Vitis HLS and validated on Catapult HLS. By providing a structured, extensible, and plug-and-play testbed, Bench4HLS establishes a foundational methodology for benchmarking LLMs in HLS workflows. 

---
# LTS-VoiceAgent: A Listen-Think-Speak Framework for Efficient Streaming Voice Interaction via Semantic Triggering and Incremental Reasoning 

**Authors**: Wenhao Zou, Yuwei Miao, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Jingwen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19952)  

**Abstract**: Real-time voice agents face a dilemma: end-to-end models often lack deep reasoning, while cascaded pipelines incur high latency by executing ASR, LLM reasoning, and TTS strictly in sequence, unlike human conversation where listeners often start thinking before the speaker finishes. Since cascaded architectures remain the dominant choice for complex tasks, existing cascaded streaming strategies attempt to reduce this latency via mechanical segmentation (e.g., fixed chunks, VAD-based splitting) or speculative generation, but they frequently either break semantic units or waste computation on predictions that must be rolled back. To address these challenges, we propose LTS-VoiceAgent, a Listen-Think-Speak framework that explicitly separates when to think from how to reason incrementally. It features a Dynamic Semantic Trigger to detect meaningful prefixes, and a Dual-Role Stream Orchestrator that coordinates a background Thinker (for state maintenance) and a foreground Speaker (for speculative solving). This parallel design enables "thinking while speaking" without blocking responses. We also introduce a Pause-and-Repair benchmark containing natural disfluencies to stress-test streaming robustness. Experiments across VERA, Spoken-MQA, BigBenchAudio, and our benchmark show that LTS-VoiceAgent achieves a stronger accuracy-latency-efficiency trade-off than serial cascaded baselines and existing streaming strategies. 

---
# STELLAR: Structure-guided LLM Assertion Retrieval and Generation for Formal Verification 

**Authors**: Saeid Rajabi, Chengmo Yang, Satwik Patnaik  

**Link**: [PDF](https://arxiv.org/pdf/2601.19903)  

**Abstract**: Formal Verification (FV) relies on high-quality SystemVerilog Assertions (SVAs), but the manual writing process is slow and error-prone. Existing LLM-based approaches either generate assertions from scratch or ignore structural patterns in hardware designs and expert-crafted assertions. This paper presents STELLAR, the first framework that guides LLM-based SVA generation with structural similarity. STELLAR represents RTL blocks as AST structural fingerprints, retrieves structurally relevant (RTL, SVA) pairs from a knowledge base, and integrates them into structure-guided prompts. Experiments show that STELLAR achieves superior syntax correctness, stylistic alignment, and functional correctness, highlighting structure-aware retrieval as a promising direction for industrial FV. 

---
