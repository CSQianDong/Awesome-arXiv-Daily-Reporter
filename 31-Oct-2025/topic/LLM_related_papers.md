# Evontree: Ontology Rule-Guided Self-Evolution of Large Language Models 

**Authors**: Mingchen Tu, Zhiqiang Liu, Juan Li, Liangyurui Liu, Junjie Wang, Lei Liang, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26683)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional capabilities across multiple domains by leveraging massive pre-training and curated fine-tuning data. However, in data-sensitive fields such as healthcare, the lack of high-quality, domain-specific training corpus hinders LLMs' adaptation for specialized applications. Meanwhile, domain experts have distilled domain wisdom into ontology rules, which formalize relationships among concepts and ensure the integrity of knowledge management repositories. Viewing LLMs as implicit repositories of human knowledge, we propose Evontree, a novel framework that leverages a small set of high-quality ontology rules to systematically extract, validate, and enhance domain knowledge within LLMs, without requiring extensive external datasets. Specifically, Evontree extracts domain ontology from raw models, detects inconsistencies using two core ontology rules, and reinforces the refined knowledge via self-distilled fine-tuning. Extensive experiments on medical QA benchmarks with Llama3-8B-Instruct and Med42-v2 demonstrate consistent outperformance over both unmodified models and leading supervised baselines, achieving up to a 3.7% improvement in accuracy. These results confirm the effectiveness, efficiency, and robustness of our approach for low-resource domain adaptation of LLMs. 

---
# SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding 

**Authors**: Yiqiao Jin, Rachneet Kaur, Zhen Zeng, Sumitra Ganesh, Srijan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.26615)  

**Abstract**: Multi-page visual documents such as manuals, brochures, presentations, and posters convey key information through layout, colors, icons, and cross-slide references. While large language models (LLMs) offer opportunities in document understanding, current systems struggle with complex, multi-page visual documents, particularly in fine-grained reasoning over elements and pages. We introduce SlideAgent, a versatile agentic framework for understanding multi-modal, multi-page, and multi-layout documents, especially slide decks. SlideAgent employs specialized agents and decomposes reasoning into three specialized levels-global, page, and element-to construct a structured, query-agnostic representation that captures both overarching themes and detailed visual or textual cues. During inference, SlideAgent selectively activates specialized agents for multi-level reasoning and integrates their outputs into coherent, context-aware answers. Extensive experiments show that SlideAgent achieves significant improvement over both proprietary (+7.9 overall) and open-source models (+9.8 overall). 

---
# Encoder-Decoder or Decoder-Only? Revisiting Encoder-Decoder Large Language Model 

**Authors**: Biao Zhang, Yong Cheng, Siamak Shakeri, Xinyi Wang, Min Ma, Orhan Firat  

**Link**: [PDF](https://arxiv.org/pdf/2510.26622)  

**Abstract**: Recent large language model (LLM) research has undergone an architectural shift from encoder-decoder modeling to nowadays the dominant decoder-only modeling. This rapid transition, however, comes without a rigorous comparative analysis especially \textit{from the scaling perspective}, raising concerns that the potential of encoder-decoder models may have been overlooked. To fill this gap, we revisit encoder-decoder LLM (RedLLM), enhancing it with recent recipes from decoder-only LLM (DecLLM). We conduct a comprehensive comparison between RedLLM, pretrained with prefix language modeling (LM), and DecLLM, pretrained with causal LM, at different model scales, ranging from $\sim$150M to $\sim$8B. Using RedPajama V1 (1.6T tokens) for pretraining and FLAN for instruction tuning, our experiments show that RedLLM produces compelling scaling properties and surprisingly strong performance. While DecLLM is overall more compute-optimal during pretraining, RedLLM demonstrates comparable scaling and context length extrapolation capabilities. After instruction tuning, RedLLM achieves comparable and even better results on various downstream tasks while enjoying substantially better inference efficiency. We hope our findings could inspire more efforts on re-examining RedLLM, unlocking its potential for developing powerful and efficient LLMs. 

---
# AMO-Bench: Large Language Models Still Struggle in High School Math Competitions 

**Authors**: Shengnan An, Xunliang Cai, Xuezhi Cao, Xiaoyu Li, Yehao Lin, Junlin Liu, Xinxuan Lv, Dan Ma, Xuanlin Wang, Ziwen Wang, Shuang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.26768)  

**Abstract**: We present AMO-Bench, an Advanced Mathematical reasoning benchmark with Olympiad level or even higher difficulty, comprising 50 human-crafted problems. Existing benchmarks have widely leveraged high school math competitions for evaluating mathematical reasoning capabilities of large language models (LLMs). However, many existing math competitions are becoming less effective for assessing top-tier LLMs due to performance saturation (e.g., AIME24/25). To address this, AMO-Bench introduces more rigorous challenges by ensuring all 50 problems are (1) cross-validated by experts to meet at least the International Mathematical Olympiad (IMO) difficulty standards, and (2) entirely original problems to prevent potential performance leakages from data memorization. Moreover, each problem in AMO-Bench requires only a final answer rather than a proof, enabling automatic and robust grading for evaluation. Experimental results across 26 LLMs on AMO-Bench show that even the best-performing model achieves only 52.4% accuracy on AMO-Bench, with most LLMs scoring below 40%. Beyond these poor performances, our further analysis reveals a promising scaling trend with increasing test-time compute on AMO-Bench. These results highlight the significant room for improving the mathematical reasoning in current LLMs. We release AMO-Bench to facilitate further research into advancing the reasoning abilities of language models. this https URL 

---
# The End of Manual Decoding: Towards Truly End-to-End Language Models 

**Authors**: Zhichao Wang, Dongyang Ma, Xinting Huang, Deng Cai, Tian Lan, Jiahao Xu, Haitao Mi, Xiaoying Tang, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26697)  

**Abstract**: The "end-to-end" label for LLMs is a misnomer. In practice, they depend on a non-differentiable decoding process that requires laborious, hand-tuning of hyperparameters like temperature and top-p. This paper introduces AutoDeco, a novel architecture that enables truly "end-to-end" generation by learning to control its own decoding strategy. We augment the standard transformer with lightweight heads that, at each step, dynamically predict context-specific temperature and top-p values alongside the next-token logits. This approach transforms decoding into a parametric, token-level process, allowing the model to self-regulate its sampling strategy within a single forward pass.
Through extensive experiments on eight benchmarks, we demonstrate that AutoDeco not only significantly outperforms default decoding strategies but also achieves performance comparable to an oracle-tuned baseline derived from "hacking the test set"-a practical upper bound for any static method. Crucially, we uncover an emergent capability for instruction-based decoding control: the model learns to interpret natural language commands (e.g., "generate with low randomness") and adjusts its predicted temperature and top-p on a token-by-token basis, opening a new paradigm for steerable and interactive LLM decoding. 

---
# Value Drifts: Tracing Value Alignment During LLM Post-Training 

**Authors**: Mehar Bhatia, Shravan Nayak, Gaurav Kamath, Marius Mosbach, Karolina Stańczak, Vered Shwartz, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.26707)  

**Abstract**: As LLMs occupy an increasingly important role in society, they are more and more confronted with questions that require them not only to draw on their general knowledge but also to align with certain human value systems. Therefore, studying the alignment of LLMs with human values has become a crucial field of inquiry. Prior work, however, mostly focuses on evaluating the alignment of fully trained models, overlooking the training dynamics by which models learn to express human values. In this work, we investigate how and at which stage value alignment arises during the course of a model's post-training. Our analysis disentangles the effects of post-training algorithms and datasets, measuring both the magnitude and time of value drifts during training. Experimenting with Llama-3 and Qwen-3 models of different sizes and popular supervised fine-tuning (SFT) and preference optimization datasets and algorithms, we find that the SFT phase generally establishes a model's values, and subsequent preference optimization rarely re-aligns these values. Furthermore, using a synthetic preference dataset that enables controlled manipulation of values, we find that different preference optimization algorithms lead to different value alignment outcomes, even when preference data is held constant. Our findings provide actionable insights into how values are learned during post-training and help to inform data curation, as well as the selection of models and algorithms for preference optimization to improve model alignment to human values. 

---
# A Multi-agent Large Language Model Framework to Automatically Assess Performance of a Clinical AI Triage Tool 

**Authors**: Adam E. Flanders, Yifan Peng, Luciano Prevedello, Robyn Ball, Errol Colak, Prahlad Menon, George Shih, Hui-Ming Lin, Paras Lakhani  

**Link**: [PDF](https://arxiv.org/pdf/2510.26498)  

**Abstract**: Purpose: The purpose of this study was to determine if an ensemble of multiple LLM agents could be used collectively to provide a more reliable assessment of a pixel-based AI triage tool than a single LLM.
Methods: 29,766 non-contrast CT head exams from fourteen hospitals were processed by a commercial intracranial hemorrhage (ICH) AI detection tool. Radiology reports were analyzed by an ensemble of eight open-source LLM models and a HIPAA compliant internal version of GPT-4o using a single multi-shot prompt that assessed for presence of ICH. 1,726 examples were manually reviewed. Performance characteristics of the eight open-source models and consensus were compared to GPT-4o. Three ideal consensus LLM ensembles were tested for rating the performance of the triage tool.
Results: The cohort consisted of 29,766 head CTs exam-report pairs. The highest AUC performance was achieved with llama3.3:70b and GPT-4o (AUC= 0.78). The average precision was highest for Llama3.3:70b and GPT-4o (AP=0.75 & 0.76). Llama3.3:70b had the highest F1 score (0.81) and recall (0.85), greater precision (0.78), specificity (0.72), and MCC (0.57). Using MCC (95% CI) the ideal combination of LLMs were: Full-9 Ensemble 0.571 (0.552-0.591), Top-3 Ensemble 0.558 (0.537-0.579), Consensus 0.556 (0.539-0.574), and GPT4o 0.522 (0.500-0.543). No statistically significant differences were observed between Top-3, Full-9, and Consensus (p > 0.05).
Conclusion: An ensemble of medium to large sized open-source LLMs provides a more consistent and reliable method to derive a ground truth retrospective evaluation of a clinical AI triage tool over a single LLM alone. 

---
# OmniEduBench: A Comprehensive Chinese Benchmark for Evaluating Large Language Models in Education 

**Authors**: Min Zhang, Hao Chen, Hao Chen, Wenqi Zhang, Didi Zhu, Xin Lin, Bo Jiang, Aimin Zhou, Fei Wu, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26422)  

**Abstract**: With the rapid development of large language models (LLMs), various LLM-based works have been widely applied in educational fields. However, most existing LLMs and their benchmarks focus primarily on the knowledge dimension, largely neglecting the evaluation of cultivation capabilities that are essential for real-world educational scenarios. Additionally, current benchmarks are often limited to a single subject or question type, lacking sufficient diversity. This issue is particularly prominent within the Chinese context. To address this gap, we introduce OmniEduBench, a comprehensive Chinese educational benchmark. OmniEduBench consists of 24.602K high-quality question-answer pairs. The data is meticulously divided into two core dimensions: the knowledge dimension and the cultivation dimension, which contain 18.121K and 6.481K entries, respectively. Each dimension is further subdivided into 6 fine-grained categories, covering a total of 61 different subjects (41 in the knowledge and 20 in the cultivation). Furthermore, the dataset features a rich variety of question formats, including 11 common exam question types, providing a solid foundation for comprehensively evaluating LLMs' capabilities in education. Extensive experiments on 11 mainstream open-source and closed-source LLMs reveal a clear performance gap. In the knowledge dimension, only Gemini-2.5 Pro surpassed 60\% accuracy, while in the cultivation dimension, the best-performing model, QWQ, still trailed human intelligence by nearly 30\%. These results highlight the substantial room for improvement and underscore the challenges of applying LLMs in education. 

---
# On the Role of Context for Discourse Relation Classification in Scientific Writing 

**Authors**: Stephen Wan, Wei Liu, Michael Strube  

**Link**: [PDF](https://arxiv.org/pdf/2510.26354)  

**Abstract**: With the increasing use of generative Artificial Intelligence (AI) methods to support science workflows, we are interested in the use of discourse-level information to find supporting evidence for AI generated scientific claims. A first step towards this objective is to examine the task of inferring discourse structure in scientific writing.
In this work, we present a preliminary investigation of pretrained language model (PLM) and Large Language Model (LLM) approaches for Discourse Relation Classification (DRC), focusing on scientific publications, an under-studied genre for this task. We examine how context can help with the DRC task, with our experiments showing that context, as defined by discourse structure, is generally helpful. We also present an analysis of which scientific discourse relation types might benefit most from context. 

---
# Inference-Cost-Aware Dynamic Tree Construction for Efficient Inference in Large Language Models 

**Authors**: Yinrong Hong, Zhiquan Tan, Kai Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26577)  

**Abstract**: Large Language Models (LLMs) face significant inference latency challenges stemming from their autoregressive design and large size. To address this, speculative decoding emerges as a solution, enabling the simultaneous generation and validation of multiple tokens. While recent approaches like EAGLE-2 and EAGLE-3 improve speculative decoding using dynamic tree structures, they often neglect the impact of crucial system variables such as GPU devices and batch sizes.
Therefore, we introduce a new dynamic tree decoding approach called CAST that takes into account inference costs, including factors such as GPU configurations and batch sizes, to dynamically refine the tree structure. Through comprehensive experimentation across six diverse tasks and utilizing six distinct LLMs, our methodology demonstrates remarkable results, achieving speeds up to 5.2 times faster than conventional decoding methods. Moreover, it generally outperforms existing state-of-the-art techniques from 5% to 20%. 

---
# From Amateur to Master: Infusing Knowledge into LLMs via Automated Curriculum Learning 

**Authors**: Nishit Neema, Srinjoy Mukherjee, Sapan Shah, Gokul Ramakrishnan, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2510.26336)  

**Abstract**: Large Language Models (LLMs) excel at general tasks but underperform in specialized domains like economics and psychology, which require deep, principled understanding. To address this, we introduce ACER (Automated Curriculum-Enhanced Regimen) that transforms generalist models into domain experts without sacrificing their broad capabilities. ACER first synthesizes a comprehensive, textbook-style curriculum by generating a table of contents for a subject and then creating question-answer (QA) pairs guided by Bloom's taxonomy. This ensures systematic topic coverage and progressively increasing difficulty. The resulting synthetic corpus is used for continual pretraining with an interleaved curriculum schedule, aligning learning across both content and cognitive dimensions.
Experiments with Llama 3.2 (1B and 3B) show significant gains in specialized MMLU subsets. In challenging domains like microeconomics, where baselines struggle, ACER boosts accuracy by 5 percentage points. Across all target domains, we observe a consistent macro-average improvement of 3 percentage points. Notably, ACER not only prevents catastrophic forgetting but also facilitates positive cross-domain knowledge transfer, improving performance on non-target domains by 0.7 points. Beyond MMLU, ACER enhances performance on knowledge-intensive benchmarks like ARC and GPQA by over 2 absolute points, while maintaining stable performance on general reasoning tasks. Our results demonstrate that ACER offers a scalable and effective recipe for closing critical domain gaps in LLMs. 

---
# Bayesian Network Fusion of Large Language Models for Sentiment Analysis 

**Authors**: Rasoul Amirzadeh, Dhananjay Thiruvady, Fatemeh Shiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.26484)  

**Abstract**: Large language models (LLMs) continue to advance, with an increasing number of domain-specific variants tailored for specialised tasks. However, these models often lack transparency and explainability, can be costly to fine-tune, require substantial prompt engineering, yield inconsistent results across domains, and impose significant adverse environmental impact due to their high computational demands. To address these challenges, we propose the Bayesian network LLM fusion (BNLF) framework, which integrates predictions from three LLMs, including FinBERT, RoBERTa, and BERTweet, through a probabilistic mechanism for sentiment analysis. BNLF performs late fusion by modelling the sentiment predictions from multiple LLMs as probabilistic nodes within a Bayesian network. Evaluated across three human-annotated financial corpora with distinct linguistic and contextual characteristics, BNLF demonstrates consistent gains of about six percent in accuracy over the baseline LLMs, underscoring its robustness to dataset variability and the effectiveness of probabilistic fusion for interpretable sentiment classification. 

---
# Unravelling the Mechanisms of Manipulating Numbers in Language Models 

**Authors**: Michal Štefánik, Timothee Mickus, Marek Kadlčík, Bertram Højer, Michal Spiegel, Raúl Vázquez, Aman Sinha, Josef Kuchař, Philipp Mondorf  

**Link**: [PDF](https://arxiv.org/pdf/2510.26285)  

**Abstract**: Recent work has shown that different large language models (LLMs) converge to similar and accurate input embedding representations for numbers. These findings conflict with the documented propensity of LLMs to produce erroneous outputs when dealing with numeric information. In this work, we aim to explain this conflict by exploring how language models manipulate numbers and quantify the lower bounds of accuracy of these mechanisms. We find that despite surfacing errors, different language models learn interchangeable representations of numbers that are systematic, highly accurate and universal across their hidden states and the types of input contexts. This allows us to create universal probes for each LLM and to trace information -- including the causes of output errors -- to specific layers. Our results lay a fundamental understanding of how pre-trained LLMs manipulate numbers and outline the potential of more accurate probing techniques in addressed refinements of LLMs' architectures. 

---
# The Geometry of Dialogue: Graphing Language Models to Reveal Synergistic Teams for Multi-Agent Collaboration 

**Authors**: Kotaro Furuya, Yuichi Kitagawa  

**Link**: [PDF](https://arxiv.org/pdf/2510.26352)  

**Abstract**: While a multi-agent approach based on large language models (LLMs) represents a promising strategy to surpass the capabilities of single models, its success is critically dependent on synergistic team composition. However, forming optimal teams is a significant challenge, as the inherent opacity of most models obscures the internal characteristics necessary for effective collaboration. In this paper, we propose an interaction-centric framework for automatic team composition that does not require any prior knowledge including their internal architectures, training data, or task performances. Our method constructs a "language model graph" that maps relationships between models from the semantic coherence of pairwise conversations, and then applies community detection to identify synergistic model clusters. Our experiments with diverse LLMs demonstrate that the proposed method discovers functionally coherent groups that reflect their latent specializations. Priming conversations with specific topics identified synergistic teams which outperform random baselines on downstream benchmarks and achieve comparable accuracy to that of manually-curated teams based on known model specializations. Our findings provide a new basis for the automated design of collaborative multi-agent LLM teams. 

---
# Do LLMs Signal When They're Right? Evidence from Neuron Agreement 

**Authors**: Kang Chen, Yaoning Wang, Kai Xiong, Zhuoka Feng, Wenhe Sun, Haotian Chen, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.26277)  

**Abstract**: Large language models (LLMs) commonly boost reasoning via sample-evaluate-ensemble decoders, achieving label free gains without ground truth. However, prevailing strategies score candidates using only external outputs such as token probabilities, entropies, or self evaluations, and these signals can be poorly calibrated after post training. We instead analyze internal behavior based on neuron activations and uncover three findings: (1) external signals are low dimensional projections of richer internal dynamics; (2) correct responses activate substantially fewer unique neurons than incorrect ones throughout generation; and (3) activations from correct responses exhibit stronger cross sample agreement, whereas incorrect ones diverge. Motivated by these observations, we propose Neuron Agreement Decoding (NAD), an unsupervised best-of-N method that selects candidates using activation sparsity and cross sample neuron agreement, operating solely on internal signals and without requiring comparable textual outputs. NAD enables early correctness prediction within the first 32 generated tokens and supports aggressive early stopping. Across math and science benchmarks with verifiable answers, NAD matches majority voting; on open ended coding benchmarks where majority voting is inapplicable, NAD consistently outperforms Avg@64. By pruning unpromising trajectories early, NAD reduces token usage by 99% with minimal loss in generation quality, showing that internal signals provide reliable, scalable, and efficient guidance for label free ensemble decoding. 

---
# Pragmatic Theories Enhance Understanding of Implied Meanings in LLMs 

**Authors**: Takuma Sato, Seiya Kawano, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2510.26253)  

**Abstract**: The ability to accurately interpret implied meanings plays a crucial role in human communication and language use, and language models are also expected to possess this capability. This study demonstrates that providing language models with pragmatic theories as prompts is an effective in-context learning approach for tasks to understand implied meanings. Specifically, we propose an approach in which an overview of pragmatic theories, such as Gricean pragmatics and Relevance Theory, is presented as a prompt to the language model, guiding it through a step-by-step reasoning process to derive a final interpretation. Experimental results showed that, compared to the baseline, which prompts intermediate reasoning without presenting pragmatic theories (0-shot Chain-of-Thought), our methods enabled language models to achieve up to 9.6\% higher scores on pragmatic reasoning tasks. Furthermore, we show that even without explaining the details of pragmatic theories, merely mentioning their names in the prompt leads to a certain performance improvement (around 1-3%) in larger models compared to the baseline. 

---
# RCScore: Quantifying Response Consistency in Large Language Models 

**Authors**: Dongjun Jang, Youngchae Ahn, Hyopil Shin  

**Link**: [PDF](https://arxiv.org/pdf/2510.26193)  

**Abstract**: Current LLM evaluations often rely on a single instruction template, overlooking models' sensitivity to instruction style-a critical aspect for real-world deployments. We present RCScore, a multi-dimensional framework quantifying how instruction formulation affects model responses. By systematically transforming benchmark problems into multiple instruction styles, RCScore reveals performance variations undetected by conventional metrics. Our experiments across ten LLMs on four reasoning benchmarks demonstrate that instruction style can shift accuracy by up to 16.7% points. We introduce Cross-Response Similarity (CRS), a method applying RCScore metrics to measure stylistic self-consistency, and establish its strong correlation with task accuracy, suggesting consistency as a valuable proxy for model reliability. Additional findings show that deterministic decoding produces more stylistically stable outputs, and model scale correlates positively with cross-style consistency. RCScore offers a principled approach to assess instruction robustness. 

---
# MisSynth: Improving MISSCI Logical Fallacies Classification with Synthetic Data 

**Authors**: Mykhailo Poliakov, Nadiya Shvai  

**Link**: [PDF](https://arxiv.org/pdf/2510.26345)  

**Abstract**: Health-related misinformation is very prevalent and potentially harmful. It is difficult to identify, especially when claims distort or misinterpret scientific findings. We investigate the impact of synthetic data generation and lightweight fine-tuning techniques on the ability of large language models (LLMs) to recognize fallacious arguments using the MISSCI dataset and framework. In this work, we propose MisSynth, a pipeline that applies retrieval-augmented generation (RAG) to produce synthetic fallacy samples, which are then used to fine-tune an LLM model. Our results show substantial accuracy gains with fine-tuned models compared to vanilla baselines. For instance, the LLaMA 3.1 8B fine-tuned model achieved an over 35% F1-score absolute improvement on the MISSCI test split over its vanilla baseline. We demonstrate that introducing synthetic fallacy data to augment limited annotated resources can significantly enhance zero-shot LLM classification performance on real-world scientific misinformation tasks, even with limited computational resources. The code and synthetic dataset are available on this https URL. 

---
# On the Influence of Discourse Relations in Persuasive Texts 

**Authors**: Nawar Turk, Sevag Kaspar, Leila Kosseim  

**Link**: [PDF](https://arxiv.org/pdf/2510.26124)  

**Abstract**: This paper investigates the relationship between Persuasion Techniques (PTs) and Discourse Relations (DRs) by leveraging Large Language Models (LLMs) and prompt engineering. Since no dataset annotated with both PTs and DRs exists, we took the SemEval 2023 Task 3 dataset labelled with 19 PTs as a starting point and developed LLM-based classifiers to label each instance of the dataset with one of the 22 PDTB 3.0 level-2 DRs. In total, four LLMs were evaluated using 10 different prompts, resulting in 40 unique DR classifiers. Ensemble models using different majority-pooling strategies were used to create 5 silver datasets of instances labelled with both persuasion techniques and level-2 PDTB senses. The silver dataset sizes vary from 1,281 instances to 204 instances, depending on the majority pooling technique used. Statistical analysis of these silver datasets shows that six discourse relations (namely Cause, Purpose, Contrast, Cause+Belief, Concession, and Condition) play a crucial role in persuasive texts, especially in the use of Loaded Language, Exaggeration/Minimisation, Repetition and to cast Doubt. This insight can contribute to detecting online propaganda and misinformation, as well as to our general understanding of effective communication. 

---
# Towards Global Retrieval Augmented Generation: A Benchmark for Corpus-Level Reasoning 

**Authors**: Qi Luo, Xiaonan Li, Tingshuo Fan, Xinchi Chen, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26205)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a leading approach to reducing hallucinations in large language models (LLMs). Current RAG evaluation benchmarks primarily focus on what we call local RAG: retrieving relevant chunks from a small subset of documents to answer queries that require only localized understanding within specific text chunks. However, many real-world applications require a fundamentally different capability -- global RAG -- which involves aggregating and analyzing information across entire document collections to derive corpus-level insights (for example, "What are the top 10 most cited papers in 2023?"). In this paper, we introduce GlobalQA -- the first benchmark specifically designed to evaluate global RAG capabilities, covering four core task types: counting, extremum queries, sorting, and top-k extraction. Through systematic evaluation across different models and baselines, we find that existing RAG methods perform poorly on global tasks, with the strongest baseline achieving only 1.51 F1 score. To address these challenges, we propose GlobalRAG, a multi-tool collaborative framework that preserves structural coherence through chunk-level retrieval, incorporates LLM-driven intelligent filters to eliminate noisy documents, and integrates aggregation modules for precise symbolic computation. On the Qwen2.5-14B model, GlobalRAG achieves 6.63 F1 compared to the strongest baseline's 1.51 F1, validating the effectiveness of our method. 

---
# Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking 

**Authors**: Feng Ju, Zeyu Qin, Rui Min, Zhitao He, Lingpeng Kong, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2510.26122)  

**Abstract**: While Test-Time Scaling (TTS) has proven effective in improving the reasoning ability of large language models (LLMs), low diversity in model outputs often becomes a bottleneck; this is partly caused by the common "one problem, one solution" (1P1S) training practice, which provides a single canonical answer and can push models toward a narrow set of reasoning paths. To address this, we propose a "one problem, multiple solutions" (1PNS) training paradigm that exposes the model to a variety of valid reasoning trajectories and thus increases inference diversity. A core challenge for 1PNS is reliably measuring semantic differences between multi-step chains of thought, so we introduce Reasoning Path Divergence (RPD), a step-level metric that aligns and scores Long Chain-of-Thought solutions to capture differences in intermediate reasoning. Using RPD, we curate maximally diverse solution sets per problem and fine-tune Qwen3-4B-Base. Experiments show that RPD-selected training yields more varied outputs and higher pass@k, with an average +2.80% gain in pass@16 over a strong 1P1S baseline and a +4.99% gain on AIME24, demonstrating that 1PNS further amplifies the effectiveness of TTS. Our code is available at this https URL . 

---
# QCoder Benchmark: Bridging Language Generation and Quantum Hardware through Simulator-Based Feedback 

**Authors**: Taku Mikuriya, Tatsuya Ishigaki, Masayuki Kawarada, Shunya Minami, Tadashi Kadowaki, Yohichi Suzuki, Soshun Naito, Shunya Takata, Takumi Kato, Tamotsu Basseda, Reo Yamada, Hiroya Takamura  

**Link**: [PDF](https://arxiv.org/pdf/2510.26101)  

**Abstract**: Large language models (LLMs) have increasingly been applied to automatic programming code generation. This task can be viewed as a language generation task that bridges natural language, human knowledge, and programming logic. However, it remains underexplored in domains that require interaction with hardware devices, such as quantum programming, where human coders write Python code that is executed on a quantum computer. To address this gap, we introduce QCoder Benchmark, an evaluation framework that assesses LLMs on quantum programming with feedback from simulated hardware devices. Our benchmark offers two key features. First, it supports evaluation using a quantum simulator environment beyond conventional Python execution, allowing feedback of domain-specific metrics such as circuit depth, execution time, and error classification, which can be used to guide better generation. Second, it incorporates human-written code submissions collected from real programming contests, enabling both quantitative comparisons and qualitative analyses of LLM outputs against human-written codes. Our experiments reveal that even advanced models like GPT-4o achieve only around 18.97% accuracy, highlighting the difficulty of the benchmark. In contrast, reasoning-based models such as o3 reach up to 78% accuracy, outperforming averaged success rates of human-written codes (39.98%). We release the QCoder Benchmark dataset and public evaluation API to support further research. 

---
# Rethinking Cross-lingual Alignment: Balancing Transfer and Cultural Erasure in Multilingual LLMs 

**Authors**: HyoJung Han, Sweta Agrawal, Eleftheria Briakou  

**Link**: [PDF](https://arxiv.org/pdf/2510.26024)  

**Abstract**: Cross-lingual alignment (CLA) aims to align multilingual representations, enabling Large Language Models (LLMs) to seamlessly transfer knowledge across languages. While intuitive, we hypothesize, this pursuit of representational convergence can inadvertently cause "cultural erasure", the functional loss of providing culturally-situated responses that should diverge based on the query language. In this work, we systematically analyze this trade-off by introducing a holistic evaluation framework, the transfer-localization plane, which quantifies both desirable knowledge transfer and undesirable cultural erasure. Using this framework, we re-evaluate recent CLA approaches and find that they consistently improve factual transfer at the direct cost of cultural localization across all six languages studied. Our investigation into the internal representations of these models reveals a key insight: universal factual transfer and culturally-specific knowledge are optimally steerable at different model layers. Based on this finding, we propose Surgical Steering, a novel inference-time method that disentangles these two objectives. By applying targeted activation steering to distinct layers, our approach achieves a better balance between the two competing dimensions, effectively overcoming the limitations of current alignment techniques. 

---
# Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning 

**Authors**: Yihe Deng, I-Hung Hsu, Jun Yan, Zifeng Wang, Rujun Han, Gufeng Zhang, Yanfei Chen, Wei Wang, Tomas Pfister, Chen-Yu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.25992)  

**Abstract**: Large Language Models (LLMs) often struggle with problems that require multi-step reasoning. For small-scale open-source models, Reinforcement Learning with Verifiable Rewards (RLVR) fails when correct solutions are rarely sampled even after many attempts, while Supervised Fine-Tuning (SFT) tends to overfit long demonstrations through rigid token-by-token imitation. To address this gap, we propose Supervised Reinforcement Learning (SRL), a framework that reformulates problem solving as generating a sequence of logical "actions". SRL trains the model to generate an internal reasoning monologue before committing to each action. It provides smoother rewards based on the similarity between the model's actions and expert actions extracted from the SFT dataset in a step-wise manner. This supervision offers richer learning signals even when all rollouts are incorrect, while encouraging flexible reasoning guided by expert demonstrations. As a result, SRL enables small models to learn challenging problems previously unlearnable by SFT or RLVR. Moreover, initializing training with SRL before refining with RLVR yields the strongest overall performance. Beyond reasoning benchmarks, SRL generalizes effectively to agentic software engineering tasks, establishing it as a robust and versatile training framework for reasoning-oriented LLMs. 

---
# MossNet: Mixture of State-Space Experts is a Multi-Head Attention 

**Authors**: Shikhar Tuli, James Seale Smith, Haris Jeelani, Chi-Heng Lin, Abhishek Patel, Vasili Ramanishka, Yen-Chang Hsu, Hongxia Jin  

**Link**: [PDF](https://arxiv.org/pdf/2510.26182)  

**Abstract**: Large language models (LLMs) have significantly advanced generative applications in natural language processing (NLP). Recent trends in model architectures revolve around efficient variants of transformers or state-space/gated-recurrent models (SSMs, GRMs). However, prevailing SSM/GRM-based methods often emulate only a single attention head, potentially limiting their expressiveness. In this work, we propose MossNet, a novel mixture-of-state-space-experts architecture that emulates a linear multi-head attention (MHA). MossNet leverages a mixture-of-experts (MoE) implementation not only in channel-mixing multi-layered perceptron (MLP) blocks but also in the time-mixing SSM kernels to realize multiple "attention heads." Extensive experiments on language modeling and downstream evaluations show that MossNet outperforms both transformer- and SSM-based architectures of similar model size and data budgets. Larger variants of MossNet, trained on trillions of tokens, further confirm its scalability and superior performance. In addition, real-device profiling on a Samsung Galaxy S24 Ultra and an Nvidia A100 GPU demonstrate favorable runtime speed and resource usage compared to similarly sized baselines. Our results suggest that MossNet is a compelling new direction for efficient, high-performing recurrent LLM architectures. 

---
# SymCode: A Neurosymbolic Approach to Mathematical Reasoning via Verifiable Code Generation 

**Authors**: Sina Bagheri Nezhad, Yao Li, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2510.25975)  

**Abstract**: Large Language Models (LLMs) often struggle with complex mathematical reasoning, where prose-based generation leads to unverified and arithmetically unsound solutions. Current prompting strategies like Chain of Thought still operate within this unreliable medium, lacking a mechanism for deterministic verification. To address these limitations, we introduce SymCode, a neurosymbolic framework that reframes mathematical problem-solving as a task of verifiable code generation using the SymPy library. We evaluate SymCode on challenging benchmarks, including MATH-500 and OlympiadBench, demonstrating significant accuracy improvements of up to 13.6 percentage points over baselines. Our analysis shows that SymCode is not only more token-efficient but also fundamentally shifts model failures from opaque logical fallacies towards transparent, programmatic errors. By grounding LLM reasoning in a deterministic symbolic engine, SymCode represents a key step towards more accurate and trustworthy AI in formal domains. 

---
# Semantic Label Drift in Cross-Cultural Translation 

**Authors**: Mohsinul Kabir, Tasnim Ahmed, Md Mezbaur Rahman, Polydoros Giannouris, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2510.25967)  

**Abstract**: Machine Translation (MT) is widely employed to address resource scarcity in low-resource languages by generating synthetic data from high-resource counterparts. While sentiment preservation in translation has long been studied, a critical but underexplored factor is the role of cultural alignment between source and target languages. In this paper, we hypothesize that semantic labels are drifted or altered during MT due to cultural divergence. Through a series of experiments across culturally sensitive and neutral domains, we establish three key findings: (1) MT systems, including modern Large Language Models (LLMs), induce label drift during translation, particularly in culturally sensitive domains; (2) unlike earlier statistical MT tools, LLMs encode cultural knowledge, and leveraging this knowledge can amplify label drift; and (3) cultural similarity or dissimilarity between source and target languages is a crucial determinant of label preservation. Our findings highlight that neglecting cultural factors in MT not only undermines label fidelity but also risks misinterpretation and cultural conflict in downstream applications. 

---
# Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data 

**Authors**: Haoran Deng, Yingyu Lin, Zhenghao Lin, Xiao Liu, Yizhou Sun, Yi-An Ma, Yeyun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2510.25804)  

**Abstract**: Long-context language models unlock advanced capabilities in reasoning, code generation, and document summarization by leveraging dependencies across extended spans of text. However, a significant portion of readily available long-text data lacks meaningful long-distance dependencies; most spans can be predicted using only local context. Training on such data is inefficient, making careful data selection crucial. Therefore, we introduce LongFilter, a framework for curating training data tailored to long-context pretraining. LongFilter measures the information gain provided by extended context by contrasting model predictions under long-context versus short-context settings, thereby identifying samples where long-range dependencies are essential. Experiments with LLaMA-3-8B, extending its context length from 8K to 64K, show that LongFilter efficiently selects high-quality data and yields substantial improvements on benchmarks such as HELMET, LongBench, and RULER. 

---
# LISTEN to Your Preferences: An LLM Framework for Multi-Objective Selection 

**Authors**: Adam S. Jovine, Tinghan Ye, Francis Bahk, Jingjing Wang, David B. Shmoys, Peter I. Frazier  

**Link**: [PDF](https://arxiv.org/pdf/2510.25799)  

**Abstract**: Human experts often struggle to select the best option from a large set of items with multiple competing objectives, a process bottlenecked by the difficulty of formalizing complex, implicit preferences. To address this, we introduce LISTEN, a framework that leverages a Large Language Model (LLM) as a zero-shot preference oracle, guided only by an expert's high-level priorities in natural language. To operate within LLM constraints like context windows and inference costs, we propose two iterative algorithms: LISTEN-U, which uses the LLM to refine a parametric utility function, and LISTEN-T, a non-parametric method that performs tournament-style selections over small batches of solutions. Evaluated on diverse tasks including flight booking, shopping, and exam scheduling, our results show LISTEN-U excels when preferences are parametrically aligned (a property we measure with a novel concordance metric), while LISTEN-T offers more robust performance. This work explores a promising direction for steering complex multi-objective decisions directly with natural language, reducing the cognitive burden of traditional preference elicitation. 

---
# PORTool: Tool-Use LLM Training with Rewarded Tree 

**Authors**: Feijie Wu, Weiwu Zhu, Yuxiang Zhang, Soumya Chatterjee, Jiarong Zhu, Fan Mo, Rodin Luo, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.26020)  

**Abstract**: Current tool-use large language models (LLMs) are trained on static datasets, enabling them to interact with external tools and perform multi-step, tool-integrated reasoning, which produces tool-call trajectories. However, these models imitate how a query is resolved in a generic tool-call routine, thereby failing to explore possible solutions and demonstrating limited performance in an evolved, dynamic tool-call environment. In this work, we propose PORTool, a reinforcement learning (RL) method that encourages a tool-use LLM to explore various trajectories yielding the correct answer. Specifically, this method starts with generating multiple rollouts for a given query, and some of them share the first few tool-call steps, thereby forming a tree-like structure. Next, we assign rewards to each step, based on its ability to produce a correct answer and make successful tool calls. A shared step across different trajectories receives the same reward, while different steps under the same fork receive different rewards. Finally, these step-wise rewards are used to calculate fork-relative advantages, blended with trajectory-relative advantages, to train the LLM for tool use. The experiments utilize 17 tools to address user queries, covering both time-sensitive and time-invariant topics. We conduct ablation studies to systematically justify the necessity and the design robustness of step-wise rewards. Furthermore, we compare the proposed PORTool with other training approaches and demonstrate significant improvements in final accuracy and the number of tool-call steps. 

---
# StreetMath: Study of LLMs' Approximation Behaviors 

**Authors**: Chiung-Yi Tseng, Somshubhra Roy, Maisha Thasin, Danyang Zhang, Blessing Effiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.25776)  

**Abstract**: There is a substantial body of literature examining the mathematical reasoning capabilities of large language models (LLMs), particularly their performance on precise arithmetic operations in autoregressive architectures. However, their ability to perform approximate reasoning in informal, fast-paced mathematical operations has received far less attention, especially among non-autoregressive decoder models. Our work addresses this gap by introducing StreetMath, a benchmark designed to evaluate models' approximation abilities under real-world approximation scenarios. We conduct extensive evaluations across different LLM architectures: Qwen3-4B-Instruct-2507, Qwen3-4B-Thinking-2507, Dream-v0-Instruct-7B, Falcon-Mamba-7B-Instruct, and Mamba-GPT-3B. Furthermore, we apply mechanistic interpretability techniques to probe their internal computational states. Our analysis reveals that LLMs generally attempt to compute exact values or invoke external tools even in tasks that call for approximation. Moreover, while models sometimes reach the correct answer in early layers or steps, they still consume more tokens when solving approximation tasks. Additional experiments indicate that exact and approximate arithmetic operations rely on largely separate neural components. Drawing upon research on cognitive psychology, we argue that LLMs do not exhibit cognitive miserliness in the same way humans do in street math settings. We open source our work this https URL 

---
# Defeating the Training-Inference Mismatch via FP16 

**Authors**: Penghui Qi, Zichen Liu, Xiangxin Zhou, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.26788)  

**Abstract**: Reinforcement learning (RL) fine-tuning of large language models (LLMs) often suffers from instability due to the numerical mismatch between the training and inference policies. While prior work has attempted to mitigate this issue through algorithmic corrections or engineering alignments, we show that its root cause lies in the floating point precision itself. The widely adopted BF16, despite its large dynamic range, introduces large rounding errors that breaks the consistency between training and inference. In this work, we demonstrate that simply reverting to \textbf{FP16} effectively eliminates this mismatch. The change is simple, fully supported by modern frameworks with only a few lines of code change, and requires no modification to the model architecture or learning algorithm. Our results suggest that using FP16 uniformly yields more stable optimization, faster convergence, and stronger performance across diverse tasks, algorithms and frameworks. We hope these findings motivate a broader reconsideration of precision trade-offs in RL fine-tuning. 

---
# Ideology-Based LLMs for Content Moderation 

**Authors**: Stefano Civelli, Pietro Bernardelle, Nardiena A. Pratama, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2510.25805)  

**Abstract**: Large language models (LLMs) are increasingly used in content moderation systems, where ensuring fairness and neutrality is essential. In this study, we examine how persona adoption influences the consistency and fairness of harmful content classification across different LLM architectures, model sizes, and content modalities (language vs. vision). At first glance, headline performance metrics suggest that personas have little impact on overall classification accuracy. However, a closer analysis reveals important behavioral shifts. Personas with different ideological leanings display distinct propensities to label content as harmful, showing that the lens through which a model "views" input can subtly shape its judgments. Further agreement analyses highlight that models, particularly larger ones, tend to align more closely with personas from the same political ideology, strengthening within-ideology consistency while widening divergence across ideological groups. To show this effect more directly, we conducted an additional study on a politically targeted task, which confirmed that personas not only behave more coherently within their own ideology but also exhibit a tendency to defend their perspective while downplaying harmfulness in opposing views. Together, these findings highlight how persona conditioning can introduce subtle ideological biases into LLM outputs, raising concerns about the use of AI systems that may reinforce partisan perspectives under the guise of neutrality. 

---
# Normative Reasoning in Large Language Models: A Comparative Benchmark from Logical and Modal Perspectives 

**Authors**: Kentaro Ozeki, Risako Ando, Takanobu Morishita, Hirohiko Abe, Koji Mineshima, Mitsuhiro Okada  

**Link**: [PDF](https://arxiv.org/pdf/2510.26606)  

**Abstract**: Normative reasoning is a type of reasoning that involves normative or deontic modality, such as obligation and permission. While large language models (LLMs) have demonstrated remarkable performance across various reasoning tasks, their ability to handle normative reasoning remains underexplored. In this paper, we systematically evaluate LLMs' reasoning capabilities in the normative domain from both logical and modal perspectives. Specifically, to assess how well LLMs reason with normative modals, we make a comparison between their reasoning with normative modals and their reasoning with epistemic modals, which share a common formal structure. To this end, we introduce a new dataset covering a wide range of formal patterns of reasoning in both normative and epistemic domains, while also incorporating non-formal cognitive factors that influence human reasoning. Our results indicate that, although LLMs generally adhere to valid reasoning patterns, they exhibit notable inconsistencies in specific types of normative reasoning and display cognitive biases similar to those observed in psychological studies of human reasoning. These findings highlight challenges in achieving logical consistency in LLMs' normative reasoning and provide insights for enhancing their reliability. All data and code are released publicly at this https URL. 

---
# SecureReviewer: Enhancing Large Language Models for Secure Code Review through Secure-aware Fine-tuning 

**Authors**: Fang Liu, Simiao Liu, Yinghao Zhu, Xiaoli Lian, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26457)  

**Abstract**: Identifying and addressing security issues during the early phase of the development lifecycle is critical for mitigating the long-term negative impacts on software systems. Code review serves as an effective practice that enables developers to check their teammates' code before integration into the codebase. To streamline the generation of review comments, various automated code review approaches have been proposed, where LLM-based methods have significantly advanced the capabilities of automated review generation. However, existing models primarily focus on general-purpose code review, their effectiveness in identifying and addressing security-related issues remains underexplored. Moreover, adapting existing code review approaches to target security issues faces substantial challenges, including data scarcity and inadequate evaluation metrics. To address these limitations, we propose SecureReviewer, a new approach designed for enhancing LLMs' ability to identify and resolve security-related issues during code review. Specifically, we first construct a dataset tailored for training and evaluating secure code review capabilities. Leveraging this dataset, we fine-tune LLMs to generate code review comments that can effectively identify security issues and provide fix suggestions with our proposed secure-aware fine-tuning strategy. To mitigate hallucination in LLMs and enhance the reliability of their outputs, we integrate the RAG technique, which grounds the generated comments in domain-specific security knowledge. Additionally, we introduce SecureBLEU, a new evaluation metric designed to assess the effectiveness of review comments in addressing security issues. Experimental results demonstrate that SecureReviewer outperforms state-of-the-art baselines in both security issue detection accuracy and the overall quality and practical utility of generated review comments. 

---
# Reasoning Curriculum: Bootstrapping Broad LLM Reasoning from Math 

**Authors**: Bo Pang, Deqian Kong, Silvio Savarese, Caiming Xiong, Yingbo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.26143)  

**Abstract**: Reinforcement learning (RL) can elicit strong reasoning in large language models (LLMs), yet most open efforts focus on math and code. We propose Reasoning Curriculum, a simple two-stage curriculum that first elicits reasoning skills in pretraining-aligned domains such as math, then adapts and refines these skills across other domains via joint RL. Stage 1 performs a brief cold start and then math-only RL with verifiable rewards to develop reasoning skills. Stage 2 runs joint RL on mixed-domain data to transfer and consolidate these skills. The curriculum is minimal and backbone-agnostic, requiring no specialized reward models beyond standard verifiability checks. Evaluated on Qwen3-4B and Llama-3.1-8B over a multi-domain suite, reasoning curriculum yields consistent gains. Ablations and a cognitive-skill analysis indicate that both stages are necessary and that math-first elicitation increases cognitive behaviors important for solving complex problems. Reasoning Curriculum provides a compact, easy-to-adopt recipe for general reasoning. 

---
# SIRAJ: Diverse and Efficient Red-Teaming for LLM Agents via Distilled Structured Reasoning 

**Authors**: Kaiwen Zhou, Ahmed Elgohary, A S M Iftekhar, Amin Saied  

**Link**: [PDF](https://arxiv.org/pdf/2510.26037)  

**Abstract**: The ability of LLM agents to plan and invoke tools exposes them to new safety risks, making a comprehensive red-teaming system crucial for discovering vulnerabilities and ensuring their safe deployment. We present SIRAJ: a generic red-teaming framework for arbitrary black-box LLM agents. We employ a dynamic two-step process that starts with an agent definition and generates diverse seed test cases that cover various risk outcomes, tool-use trajectories, and risk sources. Then, it iteratively constructs and refines model-based adversarial attacks based on the execution trajectories of former attempts. To optimize the red-teaming cost, we present a model distillation approach that leverages structured forms of a teacher model's reasoning to train smaller models that are equally effective. Across diverse evaluation agent settings, our seed test case generation approach yields 2 -- 2.5x boost to the coverage of risk outcomes and tool-calling trajectories. Our distilled 8B red-teamer model improves attack success rate by 100%, surpassing the 671B Deepseek-R1 model. Our ablations and analyses validate the effectiveness of the iterative framework, structured reasoning, and the generalization of our red-teamer models. 

---
# One Model to Critique Them All: Rewarding Agentic Tool-Use via Efficient Reasoning 

**Authors**: Renhao Li, Jianhong Tu, Yang Su, Hamid Alinejad-Rokny, Derek F. Wong, Junyang Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26167)  

**Abstract**: Reward models (RMs) play a critical role in aligning large language models (LLMs) with human preferences. Yet in the domain of tool learning, the lack of RMs specifically designed for function-calling tasks has limited progress toward more capable agentic AI. We introduce ToolRM, a family of lightweight generative RMs tailored for general tool-use scenarios. To build these models, we propose a novel pipeline that constructs pairwise preference data using rule-based scoring and multidimensional sampling. This yields ToolPref-Pairwise-30K, a diverse, balanced, and challenging dataset of critique tasks that supports reinforcement learning with verifiable feedback. To evaluate tool-use RMs, we also introduce TRBench$_{BFCL}$, a benchmark built on the agentic evaluation suite BFCL. Trained on our constructed data, models from the Qwen3-4B/8B series achieve up to 14.28% higher accuracy, substantially outperforming frontier models such as Claude 4 and OpenAI o3 in pairwise reward judgments. Beyond training objectives, ToolRM generalizes to broader critique tasks, including Best-of-N sampling and self-correction. Experiments on ACEBench highlight its effectiveness and efficiency, enabling inference-time scaling and reducing output token usage by over 66%. We release data and model checkpoints to facilitate future research. 

---
# Through the Judge's Eyes: Inferred Thinking Traces Improve Reliability of LLM Raters 

**Authors**: Xingjian Zhang, Tianhong Gao, Suliang Jin, Tianhao Wang, Teng Ye, Eytan Adar, Qiaozhu Mei  

**Link**: [PDF](https://arxiv.org/pdf/2510.25860)  

**Abstract**: Large language models (LLMs) are increasingly used as raters for evaluation tasks. However, their reliability is often limited for subjective tasks, when human judgments involve subtle reasoning beyond annotation labels. Thinking traces, the reasoning behind a judgment, are highly informative but challenging to collect and curate. We present a human-LLM collaborative framework to infer thinking traces from label-only annotations. The proposed framework uses a simple and effective rejection sampling method to reconstruct these traces at scale. These inferred thinking traces are applied to two complementary tasks: (1) fine-tuning open LLM raters; and (2) synthesizing clearer annotation guidelines for proprietary LLM raters. Across multiple datasets, our methods lead to significantly improved LLM-human agreement. Additionally, the refined annotation guidelines increase agreement among different LLM models. These results suggest that LLMs can serve as practical proxies for otherwise unrevealed human thinking traces, enabling label-only corpora to be extended into thinking-trace-augmented resources that enhance the reliability of LLM raters. 

---
# ORBIT - Open Recommendation Benchmark for Reproducible Research with Hidden Tests 

**Authors**: Jingyuan He, Jiongnan Liu, Vishan Vishesh Oberoi, Bolin Wu, Mahima Jagadeesh Patel, Kangrui Mao, Chuning Shi, I-Ta Lee, Arnold Overwijk, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.26095)  

**Abstract**: Recommender systems are among the most impactful AI applications, interacting with billions of users every day, guiding them to relevant products, services, or information tailored to their preferences. However, the research and development of recommender systems are hindered by existing datasets that fail to capture realistic user behaviors and inconsistent evaluation settings that lead to ambiguous conclusions. This paper introduces the Open Recommendation Benchmark for Reproducible Research with HIdden Tests (ORBIT), a unified benchmark for consistent and realistic evaluation of recommendation models. ORBIT offers a standardized evaluation framework of public datasets with reproducible splits and transparent settings for its public leaderboard. Additionally, ORBIT introduces a new webpage recommendation task, ClueWeb-Reco, featuring web browsing sequences from 87 million public, high-quality webpages. ClueWeb-Reco is a synthetic dataset derived from real, user-consented, and privacy-guaranteed browsing data. It aligns with modern recommendation scenarios and is reserved as the hidden test part of our leaderboard to challenge recommendation models' generalization ability. ORBIT measures 12 representative recommendation models on its public benchmark and introduces a prompted LLM baseline on the ClueWeb-Reco hidden test. Our benchmark results reflect general improvements of recommender systems on the public datasets, with variable individual performances. The results on the hidden test reveal the limitations of existing approaches in large-scale webpage recommendation and highlight the potential for improvements with LLM integrations. ORBIT benchmark, leaderboard, and codebase are available at this https URL. 

---
# Agentic AI Home Energy Management System: A Large Language Model Framework for Residential Load Scheduling 

**Authors**: Reda El Makroum, Sebastian Zwickl-Bernhard, Lukas Kranzl  

**Link**: [PDF](https://arxiv.org/pdf/2510.26603)  

**Abstract**: The electricity sector transition requires substantial increases in residential demand response capacity, yet Home Energy Management Systems (HEMS) adoption remains limited by user interaction barriers requiring translation of everyday preferences into technical parameters. While large language models have been applied to energy systems as code generators and parameter extractors, no existing implementation deploys LLMs as autonomous coordinators managing the complete workflow from natural language input to multi-appliance scheduling. This paper presents an agentic AI HEMS where LLMs autonomously coordinate multi-appliance scheduling from natural language requests to device control, achieving optimal scheduling without example demonstrations. A hierarchical architecture combining one orchestrator with three specialist agents uses the ReAct pattern for iterative reasoning, enabling dynamic coordination without hardcoded workflows while integrating Google Calendar for context-aware deadline extraction. Evaluation across three open-source models using real Austrian day-ahead electricity prices reveals substantial capability differences. Llama-3.3-70B successfully coordinates all appliances across all scenarios to match cost-optimal benchmarks computed via mixed-integer linear programming, while other models achieve perfect single-appliance performance but struggle to coordinate all appliances simultaneously. Progressive prompt engineering experiments demonstrate that analytical query handling without explicit guidance remains unreliable despite models' general reasoning capabilities. We open-source the complete system including orchestration logic, agent prompts, tools, and web interfaces to enable reproducibility, extension, and future research. 

---
# Autograder+: A Multi-Faceted AI Framework for Rich Pedagogical Feedback in Programming Education 

**Authors**: Vikrant Sahu, Gagan Raj Gupta, Raghav Borikar, Nitin Mane  

**Link**: [PDF](https://arxiv.org/pdf/2510.26402)  

**Abstract**: The rapid growth of programming education has outpaced traditional assessment tools, leaving faculty with limited means to provide meaningful, scalable feedback. Conventional autograders, while efficient, act as black-box systems that simply return pass/fail results, offering little insight into student thinking or learning needs.
Autograder+ is designed to shift autograding from a purely summative process to a formative learning experience. It introduces two key capabilities: automated feedback generation using a fine-tuned Large Language Model, and visualization of student code submissions to uncover learning patterns. The model is fine-tuned on curated student code and expert feedback to ensure pedagogically aligned, context-aware guidance.
In evaluation across 600 student submissions from multiple programming tasks, the system produced feedback with strong semantic alignment to instructor comments. For visualization, contrastively learned code embeddings trained on 1,000 annotated submissions enable grouping solutions into meaningful clusters based on functionality and approach. The system also supports prompt-pooling, allowing instructors to guide feedback style through selected prompt templates.
By integrating AI-driven feedback, semantic clustering, and interactive visualization, Autograder+ reduces instructor workload while supporting targeted instruction and promoting stronger learning outcomes. 

---
# Scales++: Compute Efficient Evaluation Subset Selection with Cognitive Scales Embeddings 

**Authors**: Andrew M. Bean, Nabeel Seedat, Shengzhuang Chen, Jonathan Richard Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2510.26384)  

**Abstract**: The prohibitive cost of evaluating large language models (LLMs) on comprehensive benchmarks necessitates the creation of small yet representative data subsets (i.e., tiny benchmarks) that enable efficient assessment while retaining predictive fidelity. Current methods for this task operate under a model-centric paradigm, selecting benchmarking items based on the collective performance of existing models. Such approaches are limited by large upfront costs, an inability to immediately handle new benchmarks (`cold-start'), and the fragile assumption that future models will share the failure patterns of their predecessors. In this work, we challenge this paradigm and propose a item-centric approach to benchmark subset selection, arguing that selection should be based on the intrinsic properties of the task items themselves, rather than on model-specific failure patterns. We instantiate this item-centric efficient benchmarking approach via a novel method, Scales++, where data selection is based on the cognitive demands of the benchmark samples. Empirically, we show Scales++ reduces the upfront selection cost by over 18x while achieving competitive predictive fidelity. On the Open LLM Leaderboard, using just a 0.5\% data subset, we predict full benchmark scores with a 2.9% mean absolute error. We demonstrate that this item-centric approach enables more efficient model evaluation without significant fidelity degradation, while also providing better cold-start performance and more interpretable benchmarking. 

---
# Chain-of-Thought Hijacking 

**Authors**: Jianli Zhao, Tingchen Fu, Rylan Schaeffer, Mrinank Sharma, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2510.26418)  

**Abstract**: Large reasoning models (LRMs) achieve higher task performance by allocating more inference-time compute, and prior works suggest this scaled reasoning may also strengthen safety by improving refusal. Yet we find the opposite: the same reasoning can be used to bypass safeguards. We introduce Chain-of-Thought Hijacking, a jailbreak attack on reasoning models. The attack pads harmful requests with long sequences of harmless puzzle reasoning. Across HarmBench, CoT Hijacking reaches a 99%, 94%, 100%, and 94% attack success rate (ASR) on Gemini 2.5 Pro, GPT o4 mini, Grok 3 mini, and Claude 4 Sonnet, respectively - far exceeding prior jailbreak methods for LRMs. To understand the effectiveness of our attack, we turn to a mechanistic analysis, which shows that mid layers encode the strength of safety checking, while late layers encode the verification outcome. Long benign CoT dilutes both signals by shifting attention away from harmful tokens. Targeted ablations of attention heads identified by this analysis causally decrease refusal, confirming their role in a safety subnetwork. These results show that the most interpretable form of reasoning - explicit CoT - can itself become a jailbreak vector when combined with final-answer cues. We release prompts, outputs, and judge decisions to facilitate replication. 

---
# LLMs Process Lists With General Filter Heads 

**Authors**: Arnab Sen Sharma, Giordano Rogers, Natalie Shapira, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2510.26784)  

**Abstract**: We investigate the mechanisms underlying a range of list-processing tasks in LLMs, and we find that LLMs have learned to encode a compact, causal representation of a general filtering operation that mirrors the generic "filter" function of functional programming. Using causal mediation analysis on a diverse set of list-processing tasks, we find that a small number of attention heads, which we dub filter heads, encode a compact representation of the filtering predicate in their query states at certain tokens. We demonstrate that this predicate representation is general and portable: it can be extracted and reapplied to execute the same filtering operation on different collections, presented in different formats, languages, or even in tasks. However, we also identify situations where transformer LMs can exploit a different strategy for filtering: eagerly evaluating if an item satisfies the predicate and storing this intermediate result as a flag directly in the item representations. Our results reveal that transformer LMs can develop human-interpretable implementations of abstract computational operations that generalize in ways that are surprisingly similar to strategies used in traditional functional programming patterns. 

---
# Who Has The Final Say? Conformity Dynamics in ChatGPT's Selections 

**Authors**: Clarissa Sabrina Arlinghaus, Tristan Kenneweg, Barbara Hammer, Günter W. Maier  

**Link**: [PDF](https://arxiv.org/pdf/2510.26481)  

**Abstract**: Large language models (LLMs) such as ChatGPT are increasingly integrated into high-stakes decision-making, yet little is known about their susceptibility to social influence. We conducted three preregistered conformity experiments with GPT-4o in a hiring context. In a baseline study, GPT consistently favored the same candidate (Profile C), reported moderate expertise (M = 3.01) and high certainty (M = 3.89), and rarely changed its choice. In Study 1 (GPT + 8), GPT faced unanimous opposition from eight simulated partners and almost always conformed (99.9%), reporting lower certainty and significantly elevated self-reported informational and normative conformity (p < .001). In Study 2 (GPT + 1), GPT interacted with a single partner and still conformed in 40.2% of disagreement trials, reporting less certainty and more normative conformity. Across studies, results demonstrate that GPT does not act as an independent observer but adapts to perceived social consensus. These findings highlight risks of treating LLMs as neutral decision aids and underline the need to elicit AI judgments prior to exposing them to human opinions. 

---
# LINK-KG: LLM-Driven Coreference-Resolved Knowledge Graphs for Human Smuggling Networks 

**Authors**: Dipak Meher, Carlotta Domeniconi, Guadalupe Correa-Cabrera  

**Link**: [PDF](https://arxiv.org/pdf/2510.26486)  

**Abstract**: Human smuggling networks are complex and constantly evolving, making them difficult to analyze comprehensively. Legal case documents offer rich factual and procedural insights into these networks but are often long, unstructured, and filled with ambiguous or shifting references, posing significant challenges for automated knowledge graph (KG) construction. Existing methods either overlook coreference resolution or fail to scale beyond short text spans, leading to fragmented graphs and inconsistent entity linking. We propose LINK-KG, a modular framework that integrates a three-stage, LLM-guided coreference resolution pipeline with downstream KG extraction. At the core of our approach is a type-specific Prompt Cache, which consistently tracks and resolves references across document chunks, enabling clean and disambiguated narratives for structured knowledge graph construction from both short and long legal texts. LINK-KG reduces average node duplication by 45.21% and noisy nodes by 32.22% compared to baseline methods, resulting in cleaner and more coherent graph structures. These improvements establish LINK-KG as a strong foundation for analyzing complex criminal networks. 

---
# Questionnaire meets LLM: A Benchmark and Empirical Study of Structural Skills for Understanding Questions and Responses 

**Authors**: Duc-Hai Nguyen, Vijayakumar Nanjappan, Barry O'Sullivan, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.26238)  

**Abstract**: Millions of people take surveys every day, from market polls and academic studies to medical questionnaires and customer feedback forms. These datasets capture valuable insights, but their scale and structure present a unique challenge for large language models (LLMs), which otherwise excel at few-shot reasoning over open-ended text. Yet, their ability to process questionnaire data or lists of questions crossed with hundreds of respondent rows remains underexplored. Current retrieval and survey analysis tools (e.g., Qualtrics, SPSS, REDCap) are typically designed for humans in the workflow, limiting such data integration with LLM and AI-empowered automation. This gap leaves scientists, surveyors, and everyday users without evidence-based guidance on how to best represent questionnaires for LLM consumption. We address this by introducing QASU (Questionnaire Analysis and Structural Understanding), a benchmark that probes six structural skills, including answer lookup, respondent count, and multi-hop inference, across six serialization formats and multiple prompt strategies. Experiments on contemporary LLMs show that choosing an effective format and prompt combination can improve accuracy by up to 8.8% points compared to suboptimal formats. For specific tasks, carefully adding a lightweight structural hint through self-augmented prompting can yield further improvements of 3-4% points on average. By systematically isolating format and prompting effects, our open source benchmark offers a simple yet versatile foundation for advancing both research and real-world practice in LLM-based questionnaire analysis. 

---
# The FM Agent 

**Authors**: Annan Li, Chufan Wu, Zengle Ge, Yee Hin Chong, Zhinan Hou, Lizhe Cao, Cheng Ju, Jianmin Wu, Huaiming Li, Haobo Zhang, Shenghao Feng, Mo Zhao, Fengzhi Qiu, Rui Yang, Mengmeng Zhang, Wenyi Zhu, Yingying Sun, Quan Sun, Shunhao Yan, Danyu Liu, Dawei Yin, Dou Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.26144)  

**Abstract**: Large language models (LLMs) are catalyzing the development of autonomous AI research agents for scientific and engineering discovery. We present FM Agent, a novel and general-purpose multi-agent framework that leverages a synergistic combination of LLM-based reasoning and large-scale evolutionary search to address complex real-world challenges. The core of FM Agent integrates several key innovations: 1) a cold-start initialization phase incorporating expert guidance, 2) a novel evolutionary sampling strategy for iterative optimization, 3) domain-specific evaluators that combine correctness, effectiveness, and LLM-supervised feedback, and 4) a distributed, asynchronous execution infrastructure built on Ray. Demonstrating broad applicability, our system has been evaluated across diverse domains, including operations research, machine learning, GPU kernel optimization, and classical mathematical problems. FM Agent reaches state-of-the-art results autonomously, without human interpretation or tuning -- 1976.3 on ALE-Bench (+5.2\%), 43.56\% on MLE-Bench (+4.0pp), up to 20x speedups on KernelBench, and establishes new state-of-the-art(SOTA) results on several classical mathematical problems. Beyond academic benchmarks, FM Agent shows considerable promise for both large-scale enterprise R\&D workflows and fundamental scientific research, where it can accelerate innovation, automate complex discovery processes, and deliver substantial engineering and scientific advances with broader societal impact. 

---
# Large Language Model-assisted Autonomous Vehicle Recovery from Immobilization 

**Authors**: Zhipeng Bao, Qianwen Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.26023)  

**Abstract**: Despite significant advancements in recent decades, autonomous vehicles (AVs) continue to face challenges in navigating certain traffic scenarios where human drivers excel. In such situations, AVs often become immobilized, disrupting overall traffic flow. Current recovery solutions, such as remote intervention (which is costly and inefficient) and manual takeover (which excludes non-drivers and limits AV accessibility), are inadequate. This paper introduces StuckSolver, a novel Large Language Model (LLM) driven recovery framework that enables AVs to resolve immobilization scenarios through self-reasoning and/or passenger-guided decision-making. StuckSolver is designed as a plug-in add-on module that operates on top of the AV's existing perception-planning-control stack, requiring no modification to its internal architecture. Instead, it interfaces with standard sensor data streams to detect immobilization states, interpret environmental context, and generate high-level recovery commands that can be executed by the AV's native planner. We evaluate StuckSolver on the Bench2Drive benchmark and in custom-designed uncertainty scenarios. Results show that StuckSolver achieves near-state-of-the-art performance through autonomous self-reasoning alone and exhibits further improvements when passenger guidance is incorporated. 

---
# GraphCompliance: Aligning Policy and Context Graphs for LLM-Based Regulatory Compliance 

**Authors**: Jiseong Chung, Ronny Ko, Wonchul Yoo, Makoto Onizuka, Sungmok Kim, Tae-Wan Kim, Won-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2510.26309)  

**Abstract**: Compliance at web scale poses practical challenges: each request may require a regulatory assessment. Regulatory texts (e.g., the General Data Protection Regulation, GDPR) are cross-referential and normative, while runtime contexts are expressed in unstructured natural language. This setting motivates us to align semantic information in unstructured text with the structured, normative elements of regulations. To this end, we introduce GraphCompliance, a framework that represents regulatory texts as a Policy Graph and runtime contexts as a Context Graph, and aligns them. In this formulation, the policy graph encodes normative structure and cross-references, whereas the context graph formalizes events as subject-action-object (SAO) and entity-relation triples. This alignment anchors the reasoning of a judge large language model (LLM) in structured information and helps reduce the burden of regulatory interpretation and event parsing, enabling a focus on the core reasoning step. In experiments on 300 GDPR-derived real-world scenarios spanning five evaluation tasks, GraphCompliance yields 4.1-7.2 percentage points (pp) higher micro-F1 than LLM-only and RAG baselines, with fewer under- and over-predictions, resulting in higher recall and lower false positive rates. Ablation studies indicate contributions from each graph component, suggesting that structured representations and a judge LLM are complementary for normative reasoning. 

---
# Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles 

**Authors**: Xinhang Li, Qing Guo, Junyu Chen, Zheng Guo, Shengzhe Xu, Lei Li, Lin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26242)  

**Abstract**: With increasing urban traffic complexity, Traffic Signal Control (TSC) is essential for optimizing traffic flow and improving road safety. Large Language Models (LLMs) emerge as promising approaches for TSC. However, they are prone to hallucinations in emergencies, leading to unreliable decisions that may cause substantial delays for emergency vehicles. Moreover, diverse intersection types present substantial challenges for traffic state encoding and cross-intersection training, limiting generalization across heterogeneous intersections. Therefore, this paper proposes Retrieval Augmented Generation (RAG)-enhanced distributed LLM agents with Emergency response for Generalizable TSC (REG-TSC). Firstly, this paper presents an emergency-aware reasoning framework, which dynamically adjusts reasoning depth based on the emergency scenario and is equipped with a novel Reviewer-based Emergency RAG (RERAG) to distill specific knowledge and guidance from historical cases, enhancing the reliability and rationality of agents' emergency decisions. Secondly, this paper designs a type-agnostic traffic representation and proposes a Reward-guided Reinforced Refinement (R3) for heterogeneous intersections. R3 adaptively samples training experience from diverse intersections with environment feedback-based priority and fine-tunes LLM agents with a designed reward-weighted likelihood loss, guiding REG-TSC toward high-reward policies across heterogeneous intersections. On three real-world road networks with 17 to 177 heterogeneous intersections, extensive experiments show that REG-TSC reduces travel time by 42.00%, queue length by 62.31%, and emergency vehicle waiting time by 83.16%, outperforming other state-of-the-art methods. 

---
# SciTrust 2.0: A Comprehensive Framework for Evaluating Trustworthiness of Large Language Models in Scientific Applications 

**Authors**: Emily Herron, Junqi Yin, Feiyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.25908)  

**Abstract**: Large language models (LLMs) have demonstrated transformative potential in scientific research, yet their deployment in high-stakes contexts raises significant trustworthiness concerns. Here, we introduce SciTrust 2.0, a comprehensive framework for evaluating LLM trustworthiness in scientific applications across four dimensions: truthfulness, adversarial robustness, scientific safety, and scientific ethics. Our framework incorporates novel, open-ended truthfulness benchmarks developed through a verified reflection-tuning pipeline and expert validation, alongside a novel ethics benchmark for scientific research contexts covering eight subcategories including dual-use research and bias. We evaluated seven prominent LLMs, including four science-specialized models and three general-purpose industry models, using multiple evaluation metrics including accuracy, semantic similarity measures, and LLM-based scoring. General-purpose industry models overall outperformed science-specialized models across each trustworthiness dimension, with GPT-o4-mini demonstrating superior performance in truthfulness assessments and adversarial robustness. Science-specialized models showed significant deficiencies in logical and ethical reasoning capabilities, along with concerning vulnerabilities in safety evaluations, particularly in high-risk domains such as biosecurity and chemical weapons. By open-sourcing our framework, we provide a foundation for developing more trustworthy AI systems and advancing research on model safety and ethics in scientific contexts. 

---
# Humains-Junior: A 3.8B Language Model Achieving GPT-4o-Level Factual Accuracy by Directed Exoskeleton Reasoning 

**Authors**: Nissan Yaron, Dan Bystritsky, Ben-Etzion Yaron  

**Link**: [PDF](https://arxiv.org/pdf/2510.25933)  

**Abstract**: We introduce Humans-Junior, a 3.8B model that matches GPT-4o on the FACTS Grounding public subset within a $\pm 5$ pp equivalence margin.
Results. On Q1--Q500 under identical judges, GPT-4o scores 73.5% (95% CI 69.5--77.2) and Humans-Junior 72.7% (95% CI 68.7--76.5); the paired difference is 0.8 pp (bootstrap 95% CI $-3.1$ to $+4.7$; permutation $p = 0.72$; Cohen's $d = 0.023$). TOST establishes equivalence at $\pm 5$ pp (not at $\pm 3$ pp). When purchased as managed APIs, Humans-Junior's base model (Phi-3.5-mini-instruct) is $\approx 19\times$ less expensive than GPT-4o on Microsoft AI Foundry pricing; self-hosted or edge deployments can drive incremental inference cost toward zero. Measured vs estimated pricing sources are tabulated in Appendix E.
Method. Our approach combines minimal directed "Exoskeleton Reasoning" scaffolds with behavioral fine-tuning that teaches protocol compliance (epistemic discipline) rather than domain answers. Fine-tuning alone adds little; combined, they synergize (+17.7 pp, $p < 0.001$) and reduce variance ($\approx 25\%$). In prompt-only settings on frontier models (Q1--Q100; non-comparable), directed reasoning improved GPT-4o by +11.8 pp to 85.3% and Gemini-2.5-Pro by +5.0 pp to 93.3% (baseline 88.3%, $n = 100$); see Section~5.
TL;DR. A 3.8B model achieves GPT-4o-level FACTS accuracy (equivalent within $\pm 5$ pp on Q1--Q500). Cloud pricing shows $\approx 19\times$ lower cost versus GPT-4o, and self-hosted/edge deployments can approach zero marginal cost. Pricing sources are listed in Appendix E. Frontier prompt-only gains (Q1--Q100; non-comparable) and optimized-prompt exploratory results under earlier judges are summarized in Appendix F.
Keywords: Small Language Models, Factual Grounding, Directed Reasoning, Fine-Tuning, Model Alignment, Cost-Efficient AI 

---
# Symbolically Scaffolded Play: Designing Role-Sensitive Prompts for Generative NPC Dialogue 

**Authors**: Vanessa Figueiredo, David Elumeze  

**Link**: [PDF](https://arxiv.org/pdf/2510.25820)  

**Abstract**: Large Language Models (LLMs) promise to transform interactive games by enabling non-player characters (NPCs) to sustain unscripted dialogue. Yet it remains unclear whether constrained prompts actually improve player experience. We investigate this question through The Interview, a voice-based detective game powered by GPT-4o. A within-subjects usability study ($N=10$) compared high-constraint (HCP) and low-constraint (LCP) prompts, revealing no reliable experiential differences beyond sensitivity to technical breakdowns. Guided by these findings, we redesigned the HCP into a hybrid JSON+RAG scaffold and conducted a synthetic evaluation with an LLM judge, positioned as an early-stage complement to usability testing. Results uncovered a novel pattern: scaffolding effects were role-dependent: the Interviewer (quest-giver NPC) gained stability, while suspect NPCs lost improvisational believability. These findings overturn the assumption that tighter constraints inherently enhance play. Extending fuzzy-symbolic scaffolding, we introduce \textit{Symbolically Scaffolded Play}, a framework in which symbolic structures are expressed as fuzzy, numerical boundaries that stabilize coherence where needed while preserving improvisation where surprise sustains engagement. 

---
# STaMP: Sequence Transformation and Mixed Precision for Low-Precision Activation Quantization 

**Authors**: Marco Federici, Riccardo Del Chiaro, Boris van Breugel, Paul Whatmough, Markus Nagel  

**Link**: [PDF](https://arxiv.org/pdf/2510.26771)  

**Abstract**: Quantization is the key method for reducing inference latency, power and memory footprint of generative AI models. However, accuracy often degrades sharply when activations are quantized below eight bits. Recent work suggests that invertible linear transformations (e.g. rotations) can aid quantization, by reparameterizing feature channels and weights. In this paper, we propose \textit{Sequence Transformation and Mixed Precision} (STaMP) quantization, a novel strategy that applies linear transformations along the \textit{sequence} dimension to exploit the strong local correlation in language and visual data. By keeping a small number of tokens in each intermediate activation at higher precision, we can maintain model accuracy at lower (average) activations bit-widths. We evaluate STaMP on recent LVM and LLM architectures, demonstrating that it significantly improves low bit width activation quantization and complements established activation and weight quantization methods including recent feature transformations. 

---
# Simulating and Experimenting with Social Media Mobilization Using LLM Agents 

**Authors**: Sadegh Shirani, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2510.26494)  

**Abstract**: Online social networks have transformed the ways in which political mobilization messages are disseminated, raising new questions about how peer influence operates at scale. Building on the landmark 61-million-person Facebook experiment \citep{bond201261}, we develop an agent-based simulation framework that integrates real U.S. Census demographic distributions, authentic Twitter network topology, and heterogeneous large language model (LLM) agents to examine the effect of mobilization messages on voter turnout. Each simulated agent is assigned demographic attributes, a personal political stance, and an LLM variant (\texttt{GPT-4.1}, \texttt{GPT-4.1-Mini}, or \texttt{GPT-4.1-Nano}) reflecting its political sophistication. Agents interact over realistic social network structures, receiving personalized feeds and dynamically updating their engagement behaviors and voting intentions. Experimental conditions replicate the informational and social mobilization treatments of the original Facebook study. Across scenarios, the simulator reproduces qualitative patterns observed in field experiments, including stronger mobilization effects under social message treatments and measurable peer spillovers. Our framework provides a controlled, reproducible environment for testing counterfactual designs and sensitivity analyses in political mobilization research, offering a bridge between high-validity field experiments and flexible computational modeling.\footnote{Code and data available at this https URL} 

---
# Test-Time Alignment of LLMs via Sampling-Based Optimal Control in pre-logit space 

**Authors**: Sekitoshi Kanai, Tsukasa Yoshida, Hiroshi Takahashi, Haru Kuroki, Kazumune Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2510.26219)  

**Abstract**: Test-time alignment of large language models (LLMs) attracts attention because fine-tuning LLMs requires high computational costs. In this paper, we propose a new test-time alignment method called adaptive importance sampling on pre-logits (AISP) on the basis of the sampling-based model predictive control with the stochastic control input. AISP applies the Gaussian perturbation into pre-logits, which are outputs of the penultimate layer, so as to maximize expected rewards with respect to the mean of the perturbation. We demonstrate that the optimal mean is obtained by importance sampling with sampled rewards. AISP outperforms best-of-n sampling in terms of rewards over the number of used samples and achieves higher rewards than other reward-based test-time alignment methods. 

---
# Hybrid LLM and Higher-Order Quantum Approximate Optimization for CSA Collateral Management 

**Authors**: Tao Jin, Stuart Florescu, Heyu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26217)  

**Abstract**: We address finance-native collateral optimization under ISDA Credit Support Annexes (CSAs), where integer lots, Schedule A haircuts, RA/MTA gating, and issuer/currency/class caps create rugged, legally bounded search spaces. We introduce a certifiable hybrid pipeline purpose-built for this domain: (i) an evidence-gated LLM that extracts CSA terms to a normalized JSON (abstain-by-default, span-cited); (ii) a quantum-inspired explorer that interleaves simulated annealing with micro higher order QAOA (HO-QAOA) on binding sub-QUBOs (subset size n <= 16, order k <= 4) to coordinate multi-asset moves across caps and RA-induced discreteness; (iii) a weighted risk-aware objective (Movement, CVaR, funding-priced overshoot) with an explicit coverage window U <= Reff+B; and (iv) CP-SAT as single arbiter to certify feasibility and gaps, including a U-cap pre-check that reports the minimal feasible buffer B*. Encoding caps/rounding as higher-order terms lets HO-QAOA target the domain couplings that defeat local swaps. On government bond datasets and multi-CSA inputs, the hybrid improves a strong classical baseline (BL-3) by 9.1%, 9.6%, and 10.7% across representative harnesses, delivering better cost-movement-tail frontiers under governance settings. We release governance grade artifacts-span citations, valuation matrix audit, weight provenance, QUBO manifests, and CP-SAT traces-to make results auditable and reproducible. 

---
# Beyond Synthetic Benchmarks: Evaluating LLM Performance on Real-World Class-Level Code Generation 

**Authors**: Musfiqur Rahman, SayedHassan Khatoonabadi, Emad Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2510.26130)  

**Abstract**: Large language models (LLMs) have advanced code generation at the function level, yet their ability to produce correct class-level implementations in authentic software projects remains poorly understood. This work introduces a novel benchmark derived from open-source repositories, comprising real-world classes divided into seen and unseen partitions to evaluate generalization under practical conditions. The evaluation examines multiple LLMs under varied input specifications, retrieval-augmented configurations, and documentation completeness levels.
Results reveal a stark performance disparity: LLMs achieve 84% to 89% correctness on established synthetic benchmarks but only 25% to 34% on real-world class tasks, with negligible differences between familiar and novel codebases. Comprehensive docstrings yield modest gains of 1% to 3% in functional accuracy, though statistical significance is rare. Retrieval-augmented generation proves most effective with partial documentation, improving correctness by 4% to 7% by supplying concrete implementation patterns absent from specifications. Error profiling identifies AttributeError, TypeError, and AssertionError as dominant failure modes (84% of cases), with synthetic tests overemphasizing assertion issues and real-world scenarios highlighting type and attribute mismatches. Retrieval augmentation reduces logical flaws but can introduce dependency conflicts.
The benchmark and analysis expose critical limitations in current LLM capabilities for class-level engineering, offering actionable insights for enhancing context modelling, documentation strategies, and retrieval integration in production code assistance tools. 

---
# Revisiting Multilingual Data Mixtures in Language Model Pretraining 

**Authors**: Negar Foroutan, Paul Teiletche, Ayush Kumar Tarun, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2510.25947)  

**Abstract**: The impact of different multilingual data mixtures in pretraining large language models (LLMs) has been a topic of ongoing debate, often raising concerns about potential trade-offs between language coverage and model performance (i.e., the curse of multilinguality). In this work, we investigate these assumptions by training 1.1B and 3B parameter LLMs on diverse multilingual corpora, varying the number of languages from 25 to 400. Our study challenges common beliefs surrounding multilingual training. First, we find that combining English and multilingual data does not necessarily degrade the in-language performance of either group, provided that languages have a sufficient number of tokens included in the pretraining corpus. Second, we observe that using English as a pivot language (i.e., a high-resource language that serves as a catalyst for multilingual generalization) yields benefits across language families, and contrary to expectations, selecting a pivot language from within a specific family does not consistently improve performance for languages within that family. Lastly, we do not observe a significant "curse of multilinguality" as the number of training languages increases in models at this scale. Our findings suggest that multilingual data, when balanced appropriately, can enhance language model capabilities without compromising performance, even in low-resource settings 

---
# WeaveRec: An LLM-Based Cross-Domain Sequential Recommendation Framework with Model Merging 

**Authors**: Min Hou, Xin Liu, Le Wu, Chenyi He, Hao Liu, Zhi Li, Xin Li, Si Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.26546)  

**Abstract**: Cross-Domain Sequential Recommendation (CDSR) seeks to improve user preference modeling by transferring knowledge from multiple domains. Despite the progress made in CDSR, most existing methods rely on overlapping users or items to establish cross-domain correlations-a requirement that rarely holds in real-world settings. The advent of large language models (LLM) and model-merging techniques appears to overcome this limitation by unifying multi-domain data without explicit overlaps. Yet, our empirical study shows that naively training an LLM on combined domains-or simply merging several domain-specific LLMs-often degrades performance relative to a model trained solely on the target domain. To address these challenges, we first experimentally investigate the cause of suboptimal performance in LLM-based cross-domain recommendation and model merging. Building on these insights, we introduce WeaveRec, which cross-trains multiple LoRA modules with source and target domain data in a weaving fashion, and fuses them via model merging. WeaveRec can be extended to multi-source domain scenarios and notably does not introduce additional inference-time cost in terms of latency or memory. Furthermore, we provide a theoretical guarantee that WeaveRec can reduce the upper bound of the expected error in the target domain. Extensive experiments on single-source, multi-source, and cross-platform cross-domain recommendation scenarios validate that WeaveRec effectively mitigates performance degradation and consistently outperforms baseline approaches in real-world recommendation tasks. 

---
# ProfOlaf: Semi-Automated Tool for Systematic Literature Reviews 

**Authors**: Martim Afonso, Nuno Saavedra, Bruno Lourenço, Alexandra Mendes, João Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2510.26750)  

**Abstract**: Systematic reviews and mapping studies are critical for synthesizing research, identifying gaps, and guiding future work, but they are often labor-intensive and time-consuming. Existing tools provide partial support for specific steps, leaving much of the process manual and error-prone. We present ProfOlaf, a semi-automated tool designed to streamline systematic reviews while maintaining methodological rigor. ProfOlaf supports iterative snowballing for article collection with human-in-the-loop filtering and uses large language models to assist in analyzing articles, extracting key topics, and answering queries about the content of papers. By combining automation with guided manual effort, ProfOlaf enhances the efficiency, quality, and reproducibility of systematic reviews across research fields. A video describing and demonstrating ProfOlaf is available at: this https URL 

---
# Vectorized Context-Aware Embeddings for GAT-Based Collaborative Filtering 

**Authors**: Danial Ebrat, Sepideh Ahmadian, Luis Rueda  

**Link**: [PDF](https://arxiv.org/pdf/2510.26461)  

**Abstract**: Recommender systems often struggle with data sparsity and cold-start scenarios, limiting their ability to provide accurate suggestions for new or infrequent users. This paper presents a Graph Attention Network (GAT) based Collaborative Filtering (CF) framework enhanced with Large Language Model (LLM) driven context aware embeddings. Specifically, we generate concise textual user profiles and unify item metadata (titles, genres, overviews) into rich textual embeddings, injecting these as initial node features in a bipartite user item graph. To further optimize ranking performance, we introduce a hybrid loss function that combines Bayesian Personalized Ranking (BPR) with a cosine similarity term and robust negative sampling, ensuring explicit negative feedback is distinguished from unobserved data. Experiments on the MovieLens 100k and 1M datasets show consistent improvements over state-of-the-art baselines in Precision, NDCG, and MAP while demonstrating robustness for users with limited interaction history. Ablation studies confirm the critical role of LLM-augmented embeddings and the cosine similarity term in capturing nuanced semantic relationships. Our approach effectively mitigates sparsity and cold-start limitations by integrating LLM-derived contextual understanding into graph-based architectures. Future directions include balancing recommendation accuracy with coverage and diversity, and introducing fairness-aware constraints and interpretability features to enhance system performance further. 

---
# ReaKase-8B: Legal Case Retrieval via Knowledge and Reasoning Representations with LLMs 

**Authors**: Yanran Tang, Ruihong Qiu, Xue Li, Zi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26178)  

**Abstract**: Legal case retrieval (LCR) is a cornerstone of real-world legal decision making, as it enables practitioners to identify precedents for a given query case. Existing approaches mainly rely on traditional lexical models and pretrained language models to encode the texts of legal cases. Yet there are rich information in the relations among different legal entities as well as the crucial reasoning process that uncovers how legal facts and legal issues can lead to judicial decisions. Such relational reasoning process reflects the distinctive characteristics of each case that can distinguish one from another, mirroring the real-world judicial process. Naturally, incorporating such information into the precise case embedding could further enhance the accuracy of case retrieval. In this paper, a novel ReaKase-8B framework is proposed to leverage extracted legal facts, legal issues, legal relation triplets and legal reasoning for effective legal case retrieval. ReaKase-8B designs an in-context legal case representation learning paradigm with a fine-tuned large language model. Extensive experiments on two benchmark datasets from COLIEE 2022 and COLIEE 2023 demonstrate that our knowledge and reasoning augmented embeddings substantially improve retrieval performance over baseline models, highlighting the potential of integrating legal reasoning into legal case retrieval systems. The code has been released on this https URL. 

---
