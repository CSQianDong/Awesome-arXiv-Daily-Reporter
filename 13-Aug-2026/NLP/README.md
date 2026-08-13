# Structural Silence: When AI Infrastructure Fails Speakers of Underrepresented Languages 

**Authors**: Avijit Roy, Proma Roy  

**Link**: [PDF](https://arxiv.org/pdf/2608.12278)  

**Abstract**: Artificial intelligence tools for education and language support are increasingly framed as scalable responses to access gaps in under-resourced communities. Yet the infrastructure underlying these tools, including training corpora, tokenization schemes, evaluation benchmarks, and deployment architectures, can systematically disadvantage speakers of underrepresented languages before a model is trained.
This paper examines these structural barriers through Bengali, one of the world's most widely spoken languages, focusing on AI-assisted education in low-connectivity environments. We identify four interlocking failures: a severe web presence gap, with Bengali accounting for less than 0.5% of global web content despite representing nearly 4% of the global population; a 67:1 training-token deficit between English and Bengali in major multilingual corpora; a tokenization penalty associated with Bengali's alphasyllabary script that compounds the data deficit through higher token fertility; and connectivity exclusion, with individual internet penetration at 36.5% in rural areas compared with 71.4% in urban areas.
These failures reflect longstanding resource-allocation decisions, institutional priorities, and design defaults that did not center underrepresented languages in mainstream AI development. We argue that dataset scarcity should be understood as a structural barrier rather than an isolated technical limitation, and that offline-first design should be treated as an equity-oriented infrastructure strategy. We conclude with directions for linguistics and AI research aimed at reducing these structural inequalities. 

---
# A Cascaded Unsupervised-Supervised NLP Pipeline for Detecting Accusatory Language in Public Procurement 

**Authors**: Bryan Torres, Daniel Riofrío, José Vega-Sánchez, Nathaly Orozco, Carla Parra, Karen Rosero, Felipe Grijalva  

**Link**: [PDF](https://arxiv.org/pdf/2608.12269)  

**Abstract**: Public procurement involves the allocation of substantial financial resources; therefore, continuous oversight through audits, controls, and monitoring mechanisms is essential. However, stakeholder comments and publicly available government data are often underutilized, despite their potential to reveal procedural irregularities. To address this gap, this paper analyzes metadata from Ecuador's Sistema Oficial de Contratación Pública (SOCE, Official Public Procurement System), with particular emphasis on participant comments generated during the pre-contractual phase. We propose a hybrid modeling framework that integrates unsupervised clustering and supervised classification within a natural language processing (NLP) pipeline to uncover latent patterns and detect potentially irregular procurement processes. Semantic embeddings are generated using Word2Vec, LLaMA, and RoBERTa, followed by Gaussian Mixture Models (GMMs) for unsupervised clustering. A supervised classification stage is then applied to identify accusatory or whistleblowing-style comments. Experimental results show that the combination of domain-trained Word2Vec embeddings, GMM-based clustering, and a Random Forest classifier achieves high precision and recall, even under severe class imbalance. These findings demonstrate that lightweight, domain-adapted NLP architectures can effectively support risk identification and enhance transparency in public procurement systems without requiring large-scale computational infrastructure. 

---
# One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL 

**Authors**: Simon Yu, Nicholas Tomlin, Marwa Abdulhai, Ximing Lu, Derek Chong, Abe Hou, Dilara Soylu, Sergey Levine, Christopher D. Manning, Weiyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2608.12253)  

**Abstract**: Multi-agent reinforcement learning for human-AI interaction typically relies on a single large language model to simulate user behavior. We show that this approach systematically fails to generalize, and trace the failure to simulator collapse: because the simulator LLM is mode-collapsed, an LLM policy trained against it overfits to narrow strategies that exploit the simulator's dominant mode, and such a policy transfers poorly to unseen simulators and real users. We formalize this collapse theoretically and propose two complementary solutions, one at inference time and one at training time. The inference-time solution, Verbalized Sampling, broadens the simulator's behavior by sampling from a verbalized response distribution, reducing mode collapse. The training-time solution, Co-Training, jointly optimizes the policy against a population of trainable simulators, preventing it from overfitting to any single simulator's mode. We validate both solutions on three multi-turn benchmarks: Persuasion for Good, $\tau^2$-bench, and CooperBench. Verbalized Sampling improves held-out success by up to 9% over single-simulator RL, and Co-Training pushes gains further to 14%; the human study shows similar gain on real users. Both solutions preserve the policy diversity that collapses under single-simulator RL. To support further work in this direction, we release SCOPE, an open-source framework for Population Co-Training multi-agent RL. More broadly, our results suggest that the diversity of the training environment, not only the policy, is critical to the generalization of multi-turn RL to real-world deployment. 

---
# Information Abundance Paradox: Long-Context Training Undermines Parametric Knowledge 

**Authors**: Arda Uzunoglu, Benjamin van Durme, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2608.12218)  

**Abstract**: Large language models are increasingly trained and deployed with long contexts that span documents, code repositories, and interaction histories. This scaling reflects the implicit assumption that training on longer contexts will only help the model by exposing it to richer evidence. We challenge this view by studying how the context window shapes a model's mode of learning, shifting it between parametric internalization and contextualization. We propose the Information Abundance Paradox, which hypothesizes that abundant relevant information in the training context can reduce the incentive to encode that information parametrically, thereby increasing reliance on context. In pretraining with long documents, increasing the context window improves language modeling, natural language understanding, and closed-book MCQA only up to an intermediate optimum, after which performance consistently declines. In supervised fine-tuning, more task-relevant train-time context improves performance with supporting context, but reduces robustness when context is absent or misleading at test time. Our analysis suggests that this behavior arises when longer context provides a lower complexity solution. Mechanistically, training with informative context shifts gradient pressure from feed-forward networks, often linked to parametric knowledge, toward attention modules, and causal interventions show that this shift increases reliance on context during inference. Overall, these findings support the Information Abundance Paradox and suggest that scaling toward near-infinite context is not simply a matter of supplying more data, even when high-quality long-context data is abundant. 

---
# Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus 

**Authors**: Zunhai Su, Bohan Sun, Xialie Zhuang, Shuibai Zhang, He Xiao, Jing Xiong, Hengyuan Zhang, Zhongzhu Zhou, Tiantian Zhang, Ngai Wong, Chuan-Wei Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2608.12149)  

**Abstract**: We present the first systematic study of Massive activations (MAs) in layer-interleaved HLA LLMs and uncover two architecture-aligned morphologies: MAs consistently spike immediately before full attention layers, forming pre-attention spikes (PAS), and can persist through intervening linear attention layers, giving rise to inter-spike plateaus (ISP). As full attention becomes denser, successive PAS become increasingly connected through ISP, ultimately recovering the stable MA morphology of full attention LLMs. We establish the recurrence of this organization across five linear attention architectures, six hybridization configurations, five data domains, and representative open-source hybrid models spanning 1.2B to 397B total parameters. Controlled pretraining of GDN-based hybrids at scales up to 1.3B shows that both morphologies emerge early and respond asymmetrically to output gating: full attention output gating strongly attenuates their absolute magnitudes without eliminating their layerwise organization, whereas removing GDN gates yields comparatively modest amplification. Mechanistically, our systematic-outlier analysis supports a shared lifecycle account governed by the timing of MA cancellation. PAS follows a localized write-sink-cancel process, while the extended persistence of ISP is consistent with delayed cancellation. At the full attention limit, this account recovers the stable MA morphology characteristic of full attention LLMs. Our code is available at this https URL. 

---
# A corpus-specific clinical RAG system matches or outperforms newer frontier LLMs on HealthBench 

**Authors**: Praveen Reddy, Charuta Mandke, Suvrankar Datta, Sarah Khan, Siddharth Reddy Anthireddy, Shitij Arora, Vishal Singh  

**Link**: [PDF](https://arxiv.org/pdf/2608.12138)  

**Abstract**: General-purpose large language models (LLMs) have recently been reported to match or exceed specialized clinical AI tools on medical benchmarks, but such comparisons draw on a narrow set of systems and on benchmarks developed largely in high-income settings. We evaluate VITA, a retrieval-augmented generation (RAG) system purpose-built for contextual knowledge retrieval in India and other low- and middle-income (LMIC) settings. VITA retrieves from a curated corpus of disease-specific guidelines, India-specific antimicrobial resistance data, national formulary constraints, and resource-limited care protocols; its architecture and corpus are proprietary, but the benchmark, the physician-written rubrics, and our full response and scoring outputs are public for independent verification. On 4,023 English-language HealthBench questions (80.5% of the benchmark), scored with a GPT-4.1 judge, VITA ranked first with 51.9% of possible rubric points, ahead of GPT-5.4 (46.1%), o4-mini (44.3%), Gemini 3.1 Pro (42.6%), and Claude Sonnet 4.6 (37.3%), and scored highest on 45.4% of questions. To test robustness to newer models and judge lineage, a 500-question subset was re-run against current-generation models (GPT-5.5, Claude Opus 4.8, Gemini 3.5 Pro, Grok 4.3) and graded by a neutral open-weight judge (DeepSeek-V4-Pro) sharing no lineage with any system tested. Here the gap narrowed to parity: VITA and GPT-5.5 were statistically indistinguishable on mean per-question score, while VITA led on points-weighted score and won the most questions. VITA's advantages in accuracy and completeness persisted under the neutral judge; its communication scores were lower. These results indicate that a purpose-built clinical RAG system remains competitive with frontier LLMs on an open benchmark, consistent with corpus specificity as a design variable that improves grounding at some cost to communication polish. 

---
# SAG: SQL-Retrieval Augmented Generation with Query-Time Dynamic Hyperedges 

**Authors**: Yuchao Wu, Junqin Li, XingCheng Liang, Yongjie Chen, Yinghao Liang, Linyuan Mo, Guanxian Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.12129)  

**Abstract**: While retrieval-augmented generation (RAG) has proven effective at giving LLMs access to external knowledge, mainstream dense-retrieval implementations remain inherently limited in handling structured constraints and multi-hop reasoning. Graph-based methods address this by constructing knowledge graphs offline, but they often fragment semantics, incur high maintenance, and complicate incremental updates. We propose SAG (SQL-Retrieval Augmented Generation), a structured retrieval architecture that organizes documents into an event-entity index without building a global knowledge graph. SAG represents each chunk as a semantically complete event paired with its entities, forming a latent hyperedge that preserves n-ary relations without decomposing them into triples. At query time, SAG treats shared entities as join keys to connect related chunks. This dynamically yields a query-scoped neighborhood of events, and yet every piece of evidence remains the original chunk throughout. Experiments on HotpotQA, 2WikiMultiHopQA, and MuSiQue show that SAG achieves the best retrieval and end-to-end QA performance on every benchmark, with gains that widen as reasoning-chain complexity increases. On MuSiQue, where multi-hop evidence chaining is most demanding, SAG reaches 80.36% Recall@5, outperforming the strongest baseline by 11.52 points. This work paves the way for knowledge infrastructure that enables LLM agents to retrieve and reason over continually growing organizational knowledge. 

---
# QV-PIC: Query-Aware Visual Position-Independent Caching for Efficient RAG Serving 

**Authors**: Yilin Liu, Rui Meng, Wangze Ni, Jianxin Yan, Heng Cao, Libin Zheng, Peng Cheng, Jinfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12121)  

**Abstract**: Retrieval-Augmented Generation (RAG) repeatedly prefills identical text chunks across queries, incurring redundant computations. Position-Independent Caching (PIC) mitigates it by reusing precomputed Key-Value (KV) across positions, but its efficiency is constrained by the large volume of text tokens. Rendering text chunks as images can compress the text into fewer visual tokens, but the rendered-image PIC suffers more severe quality degradation than the text PIC. This representation-specific gap primarily arises from contextual mismatches across independently compiled caches and the loss of fine-grained textual evidence during visual compression. Existing PIC repair methods mainly address the former through selective recomputation, but they incur online computation and cannot recover lost textual details. We propose QV-PIC, a query-aware dual-resolution PIC reuse framework guided by model-native templates. Offline, QV-PIC compiles visual caches under the model's native chat-template prefix, improving PIC quality without online recomputation. Online, it preserves global context with low resolution and restores fine-grained textual evidence within a high-resolution budget by cumulative query relevance scores, retaining the efficiency benefit of visual compression. Across six tasks, QV-PIC improves average F1 by 21.6 points over vanilla rendered-image PIC, closes the gap to vanilla text PIC, and surpasses optimized text PIC by 2.58 F1 while reducing TTFT by 17.2\%. Relative to full prefill, it cuts TTFT by 83.8%. 

---
# Structuring the Space of Perspectives 

**Authors**: Agnese Daffara, Sebastian Padó, Tanise Ceron  

**Link**: [PDF](https://arxiv.org/pdf/2608.12113)  

**Abstract**: The same event can be reported from different perspectives depending on the experiences, background, and beliefs of the writer or speaker. A variety of NLP areas engage with perspectives, spanning from text analysis to algorithm optimization. A wide range of operative concepts (such as stances, sentiment, frames, and arguments) has been used to capture perspectives in texts, however the precise relationships among those concepts remain unclear. Arguably, a deeper theoretical understanding of these concepts would empower more effective research on perspectives. In this paper, we address this gap by reviewing the space of perspectives in NLP and defining a set of properties that help distinguishing perspective-related concepts. Our analysis leads us to posit a hierarchy which organizes these concepts linearly along a single axis. Finally, we show how this principled conceptual hierarchy can help researchers navigate the field and select operationalizations of perspective that align with their specific research objectives. 

---
# Preference Tree Optimization: Enhancing Goal-Oriented Dialogue with Look-Ahead Simulations 

**Authors**: Lior Baruch, Moshe Butman, Kfir Bar, Doron Friedman  

**Link**: [PDF](https://arxiv.org/pdf/2608.12062)  

**Abstract**: Developing dialogue systems capable of engaging in multi-turn, goal-oriented conversations remains a significant challenge, especially in specialized domains with limited data. This research proposes a novel framework called Preference Tree Optimization (PTO), designed to iteratively improve agent models in such dialogue systems, by generating preference data using a method called Preference Tree with Look-Ahead. Focusing on Motivational Interviewing (MI) -- a counseling technique aimed at facilitating behavioral change -- we leverage virtual patients and an oracle evaluator to simulate conversations and generate rich preference datasets. By combining this method with Direct Preference Optimization (DPO), we aim to enhance the agent's decision-making capabilities over iterative training cycles. The proposed framework addresses data scarcity and advances the development of more nuanced and effective dialogue systems in goal-oriented domains. Experimental evaluations demonstrate that the PTO framework enhances dialogue agents' performance in goal-oriented conversations within the domain of Motivational Interviewing (MI). Models trained with PTO consistently outperformed the baseline in key metrics such as session satisfaction and working alliance. Additionally, incorporating look-ahead simulations led to improved long-term planning and more effective conversational strategies, with deeper look-ahead configurations yielding the most stable and high-scoring results. 

---
# Poly-Dialectal Neural Machine Translation System for Bangla Regional Dialects 

**Authors**: Rakib Ullah, Ruhul Islam Rahul, Tanbir Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2608.12018)  

**Abstract**: Regional dialectal variation poses a fundamental challenge to natural language processing (NLP) in Bangla, where over 240 million speakers communicate across diverse regional variants that diverge significantly from Standard Colloquial Bangla (SCB) in phonology, morphology, and lexicon. Contemporary neural machine trans- lation (NMT) architectures and large language models (LLMs) predominantly as- sume a homogeneous language distribution, resulting in severe performance degra- dation when translating low-resource regional dialects. In this work, we present a unified Poly-Dialectal Neural Machine Translation System capable of multi-directional translation across 12 Bangla regional dialects without routing through an inter- mediary standard pivot. We compile the largest multi-dialect parallel corpus for Bangla to date, comprising 51,531 non-null parallel sentence pairs across 12 di- alects, incorporating 2,500 expert-verified, bidirectional parallel sentence pairs for five previously unaddressed dialects. Evaluating sequence-to-sequence architec- tures under Weight-Decomposed Low-Rank Adaptation (DoRA), our fine-tuned BanglaT5 model achieves state-of-the-art translation performance (29.26 BLEU, 57.26 chrF++), outperforming NLLB-200 (615M) and mBART-50 (611M) while preserving morphological coherence. Furthermore, we conduct a systematic cross- dialectal transfer analysis and dataset scaling study, establishing empirical thresh- olds for low-resource dialect adaptation. Finally, we deploy the optimized INT8- quantized model as an open-access web application to promote digital inclusion for marginalized dialect communities. The complete dataset is publicly available at Mendeley Data (this https URL). 

---
# Asymptotic Risk Calibration for Selective Question Answering 

**Authors**: Shufan Lin, Sijin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2608.12008)  

**Abstract**: Large language models (LLMs) may generate fluent but incorrect answers, making uncertainty quantification important for reliable question answering. However, heuristic uncertainty scores cannot perfectly distinguish correct predictions from incorrect ones, and directly applying a fixed uncertainty threshold provides no statistical control over the error rate among accepted answers. To address this limitation, we propose A-CRC-QA, a post-hoc calibration framework for uncertainty-aware selective question answering. The proposed method reformulates selection-conditioned error control as a linear expectation constraint and applies a monotonized empirical-risk calibration procedure inspired by conformal risk control. Since the resulting instance-wise loss is generally non-monotone with respect to the acceptance threshold, our framework targets asymptotic rather than finite-sample risk control. A-CRC-QA is model-agnostic, requires no additional training, and can be combined with different uncertainty estimators. Experiments on CoQA and MedMCQA demonstrate its applicability to both open-ended and closed-ended question answering, achieving a favorable trade-off between accepted-answer reliability and answer retention compared with uncalibrated and confidence-bound-based baselines. 

---
# Benchmarking Trustworthiness of SLMs: Pre-trained vs. Compressed 

**Authors**: Haokun Lin, Kaijie Zhu, Haobo Xu, Yichen Wu, Zhichao Lu, Qingfu Zhang, Zhenan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.11981)  

**Abstract**: Small Language Models (SLMs) have emerged as a more efficient alternative to traditional Large Language Models (LLMs), offering promising potential in resource-constrained scenarios. Existing approaches to building SLMs typically follow two paths: training compact models from scratch, or compressing larger pre-trained models using methods such as pruning, quantization, or distillation. As language models become increasingly integrated into real-world applications, ensuring their trustworthiness has become a critical concern. However, how to build trustworthy SLMs remains an underexplored question. In this work, we present a comprehensive evaluation of SLM trustworthiness across multiple dimensions, including fairness, robustness, privacy, and ethics. We first examine the effects of pruning and quantization, and find that quantization is significantly more effective in preserving trustworthiness compared to pruning. More importantly, we demonstrate that compressing a reliable large model via quantization can produce SLMs with superior trustworthiness and adaptability compared to using small models trained from scratch. Furthermore, knowledge distillation from trustworthy teacher models can further enhance the reliability of SLMs. We hope our findings provide practical guidance and a foundation for future research into the development and deployment of trustworthy small language models. 

---
# Accuracy and Order Sensitivity Diverge Under Label-Free Strategies 

**Authors**: Karl Hanna, Chen Feng  

**Link**: [PDF](https://arxiv.org/pdf/2608.11947)  

**Abstract**: Multiple-choice benchmarks are widely used to evaluate large language models, but MCQ scores conflate knowledge with sensitivity to option order, which makes them unreliable measures of model knowledge. In this paper, we test whether preventing a model from seeing option labels while committing to an answer removes positional influence and, in turn, improves performance. We evaluate two different strategies for mitigating bias. The first uses a generation-then-matching approach, and the second scores options in isolation, which is positionally unbiased by construction. Neither reliably improves accuracy. A complete decomposition shows that the bottleneck is withholding options, not the matching step. The only configuration that consistently matches the baseline is the one that shows the model all options paired with an LLM matcher. However, eliminating positional influence entirely still does not reliably yield accuracy gains, while cyclic permutation often improves them. For two-stage prompting, an aggregate measure of recall imbalance and a direct per-question measure of order sensitivity both fail to show reliable debiasing. 

---
# Spark-to-Paper: End-to-End Research Paper Generation as a Composable Skill 

**Authors**: Zhuoyang Qian, Biao Wu, Yiran Wang, Chris D Yan, Desan Dai, Liangwei Zheng, Jin Jiang, Junsheng Zhang, Wenhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11924)  

**Abstract**: Turning a research idea into a complete paper requires more than text generation: the system must retrieve literature, design and execute experiments, revise claims according to evidence, produce publication-ready figures, and maintain consistency across a long generation process. We present Spark-to-Paper, an end-to-end research paper generation system implemented as thirteen composable skills inside an existing coding assistant, without requiring a separate agent platform or orchestration service. Spark-to-Paper separates model-based judgment from deterministic operations that can be directly executed and checked. It further separates experiment planning from reporting, so that required evidence is specified before results are observed and manuscript claims are revised according to measured outcomes. To improve reliability over long research trajectories, the system combines deterministic integrity checks with self-critique and bounds a failure mode we call the Self-Refutation Loop, in which repeated experiments continue to reject the original research objective. Spark-to-Paper also produces editable vector figures through programmatic plotting for experimental results and code-based reconstruction for generated method diagrams. Across eight controlled research topics, Spark-to-Paper achieves 99.5% citation validity and 96.4% figure editability. A controlled ablation increases fabrication detection from 14% for a single-pass draft to 92% with the full integrity and review stack, while adversarial review achieves 74% precision. The full system uses 11.9M tokens, costs $8.1 per manuscript, and requires 3.2 hours on average. These results show that end-to-end research paper generation can be implemented as a lightweight, composable workflow inside existing coding assistants while keeping experimental evidence central to how claims are accepted, revised, or abandoned. 

---
# LODESTAR: Trustworthy Entropy Is Navigated, Not Merely Measured -- Reinforced Polarizer Keeps a Frozen LLM from Being Confidently Misled by the Wrong Evidence 

**Authors**: Po-Jen Ko, Che-Cheng Wu, Hung-Chun Hsu, Li-Yang Chang, Chuan-Ju Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11922)  

**Abstract**: Predictive-distribution entropy makes a strong selection rule in retrieval-augmented question answering: across five QA benchmarks, keeping the candidate answer that a frozen respondent LLM produces with the lowest answer-token entropy lifts mean answer $F_1$ from 0.4769 to 0.5148 over the retriever's top-ranked passage, with no gold answers. Yet this lowest-entropy rule, which prior entropy-based selectors adopt, fails in a specific and consequential way: a misleading passage makes the respondent confidently wrong, driving its entropy down precisely where the signal looks most trustworthy. We show that the failure comes from the passage the respondent reads -- and the context that passage is read in is an input we can intervene on. We introduce LODESTAR, to our knowledge the first method to score a text intervention by the uncertainty it induces in a third-party frozen respondent, compared across one question's candidates. LODESTAR uses reinforcement learning to train, once and offline, a polarizer -- a short fixed natural-language string inserted into the respondent's prompt and never into its weights; its training labels are built offline from gold answers and two LLM judges, and inference reads neither. Evaluating every competing selector under the same frozen respondent and the same candidate pools on 5,008 questions, LODESTAR attains the highest mean $F_1$ of any inference-ready selector (0.5148 to 0.5339), the highest exact match (0.4136), and the highest GPT-4o judge score of the frozen-respondent configurations judged (0.6435); its three-seed mean wins all 70 method-by-dataset $F_1$ cells against fourteen published configurations while remaining paired-significant against every one. The gain holds both in-domain and out-of-domain, and ablating the polarizer shows it is what makes the respondent read a misleading passage less often (26.0% against 30.3%). 

---
# LazyTrain: Limited-resource Allocation toward Zero-waste Yield Optimization in Large Language Model Training 

**Authors**: Xiaojun Wu, Cehao Yang, Honghao Liu, Xueyuan Lin, Xuhui Jiang, Chengjin Xu, Jia Li, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.11919)  

**Abstract**: Training large language models on limited hardware is increasingly a scheduling problem across GPU compute, host memory, PCIe transfer, and storage bandwidth. Existing offloading systems reduce GPU residency, and MegaTrain shows that a CPU-master layer-streaming executor can train large models on a single GPU, but fixed checkpointing and placement heuristics still leave communication exposed on the critical path. We propose LazyTrain, an optimization layer over a layer-streaming executor. LazyTrain formulates checkpoint selection, activation placement, recomputation, and CPU-GPU-NVMe communication overlap as a mixed-integer scheduling problem, then executes the solved policy during training. It further couples 8-bit optimizer states with fast gradient clipping as a single Hybrid 8-bit operator: state compression reduces optimizer-state memory, while fast clipping counteracts the additional CPU-side update overhead. Across H800 experiments from Qwen2.5-3B to Qwen3.6-27B, LazyTrain improves sustained TFLOPS over matched baselines runs by approximately 1.24$\times$; RTX 3090 experiments likewise increase the maximum feasible batch size by one at each model scale. In the primary Qwen3.6-27B H800 MetaMathQA run, LazyTrain reaches 219.95 TFLOPS and 1361 tokens/s at batch size 72, peaks at 68.84\,GB of GPU memory, and obtains 95.42\% exact-match accuracy on the full evaluation split. The source code is available at this https URL. 

---
# Total Recall at What Cost? Benchmarking the Serving Cost of Agentic Memory Systems 

**Authors**: Natchanon Pollertlam, Witchayut Kornsuwannawit  

**Link**: [PDF](https://arxiv.org/pdf/2608.11879)  

**Abstract**: Long-running conversational agents increasingly rely on a memory system to avoid resending the whole conversation each turn, yet how much that costs to serve has received little systematic benchmarking. We compare three memory systems (Mem0, Hindsight, and Mastra Observational Memory) against two reference strategies -- a fixed-size rolling window and resubmitting the full transcript -- across two backbones and conversations of up to 400 turns, pairing every cost measurement with answer accuracy on 665 LoCoMo questions. First, a memory system's serving cost cannot be predicted from conversation length and message size alone: a regression that tracks the two reference strategies closely misses the memory systems by 18-69%, their cost driven instead by internal memory behavior. Second, a break-even analysis shows that whether -- and when -- a memory system becomes cheaper to serve than the full transcript is highly sensitive to the system and the backbone, from the first tens of turns for the cheapest to never within 400 turns for the most expensive. Third, no system wins on both axes: accuracy spans 21-54%, and the backbone choice drives cost as much as the memory system does. 

---
# When the Knowledge Base Becomes the Gold Standard: Measuring Resource-Shared Evaluation Loops in Entity-Level Machine Translation 

**Authors**: Jinhyung Bae, Dain Kil, Seongmin Oh, Seungmin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.11843)  

**Abstract**: The Seungjeongwon Ilgi, a UNESCO Memory of the World record, is only 37.4% translated, and the most conspicuous failure mode in automatic translation is the person name -- a misread name corrupts the historical fact rather than merely the surface. Low-resource historical domains have no expert gold standard for entity translation, so practitioners substitute a knowledge base (KB) for the gold. That KB is the same resource injected into the system: scoring becomes self-referential and the metric measures instruction compliance rather than translation quality.
We measure this loop. Using expert person-name annotations from the National Institute of Korean History as a gold independent of the injection pipeline, we hold the entity set fixed and vary only the provenance of the correct reading. Of 527 expert-annotated mentions, only 31.1% lie outside the injection pipeline, and the residual loop is not uniform -- in the overlapping segment the injected reading agrees with the human translation 97.8% of the time against 70.1% in the independent one, so the segment that looks healthiest is the one the loop is holding up.
Across four models, a difference-in-differences analysis shows the gain from KB injection is confined to the segment whose gold shares the injected resource; in the independent segment it is at or below zero. Post-injection preservation clusters in a narrow 0.910-0.996 band even though baseline capability differs fivefold, so the reported gain is the complement of prior performance and weaker models appear to improve more dramatically. On an independent sample built by removing the construction filter, the measure replicates within model (overlapping intervals) while discriminating between models (non-overlapping intervals) -- it reflects a property of the model, not of the sample. 

---
# Located but Not Releasable: Silent Gate Inversion and Bounded Linear Release 

**Authors**: Xining Xun  

**Link**: [PDF](https://arxiv.org/pdf/2608.11822)  

**Abstract**: A growing body of work reports that language models represent task-relevant latent structure that they fail to use. Whether such structure, once located, can be converted into behavior is a separate question that is rarely tested end to end. We submit the complete pipeline -- detect, localize, and release -- to a fully preregistered stress test on a 25.7M transformer trained on causal-evidence discrimination, where a known suppression phenomenon (latent causal structure present but behaviorally unused) has previously been documented. Every threshold, claim template, and decision-tree branch was hashed and archived before any corresponding data existed. Three findings. (i) Localization succeeds: interventions at observation-evidence channels of mid layers restore target behavior on otherwise-suppressed worlds (paired release advantages $0.563$ and $0.854$, 97.5% CIs excluding zero; best-site release rate $0.889$). (ii) Gating fails out of distribution: a detector calibrated to trigger on zero out-of-distribution calibration worlds triggers on 6.9-7.3% of held-out in-distribution generations and on zero of the 2,400 held-out generations that actually need it -- a complete inversion that silently reduces the gated pipeline to its base model. (iii) Linear release is capped: removing the gate and injecting a per-instance linear direction unconditionally yields a monotone dose-response that plateaus far below the preregistered release margin (intercept $0.382 \to 0.311 \to 0.264$ vs. threshold $\le 0.08$); per-instance adaptivity adds less than $\pm 0.03$. The failure is doubly located: the detector is OOD-inverted, and the entire family of linear release directions at this site and resolution is bounded away from sufficiency. The two failures are dissociable, and neither overturns localization. Every number traces to a hashed artifact in the released audit chain. 

---
# Hybrid Gated Attention 

**Authors**: Zekun Zhou, Ruobing Xie, Lanrui Wang, Weixuan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.11805)  

**Abstract**: Gated attention is an effective approach to mitigate attention sinks and enhance the representational capacity of attention. To further extend its effectiveness-efficiency Pareto frontier, we propose a Hybrid Gated Attention (HyGA) framework that contains three types of gating strategies. Specifically, these gates leverage diverse information from multiple stages of attention, and collaboratively build element-wise/head-wise gating from multiple perspectives, capturing intra-head and cross-head information interactions. Through our hybrid gating components, HyGA could provide multi-source modulation signals, enabling more comprehensive control over information flow and improving the representational capacity of attention. We also introduce low-rank matrix decomposition and learnable attention sink to further enhance training efficiency and stability. In experiments, we evaluate HyGA on widely-used benchmarks based on different backbones. The experimental results show that our HyGA comprehensively improves both training loss and various downstream performances compared with Gated attention. HyGA has also been verified to achieve the best performance at different computation costs, with comprehensive model analyses for better understanding. The proposed HyGA sheds light on a more effective, efficient, and stable attention mechanism. 

---
# TELLME: Test-Enhanced Learning for Language Model Enrichment 

**Authors**: Minjun Kim, Inho Won, Hyeonseok Lim, MinKyu Kim, Junghun Yuk, Wooyoung Go, Jongyoul Park, Jungyeul Park, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2608.11788)  

**Abstract**: Continual pre-training (CPT) has been widely adopted as a method for domain adaptation in large language models. However, CPT has consistently been accompanied by challenges, such as the difficulty of acquiring large-scale domain-specific datasets and high computational costs. In this study, we propose a novel method called Test-Enhanced Learning for Language Model Enrichment (TELLME) to alleviate these issues. TELLME leverages the TestEnhanced Learning (TEL) principle, whereby the model's training efficiency is improved using quizzes during training. It integrates this principle with CPT, thereby promoting efficient domain-specific knowledge acquisition and long-term memory retention. Experimental results demonstrate that TELLME outperforms existing methods by up to 23.6% in the financial domain and achieves a 9.8% improvement in long-term memory retention. 

---
# GRPO for Financial Advice Generation: Outperforming Commercial LLMs under CATE Evaluation 

**Authors**: Ofir Ben Shoham, Shrutendra Harsola, Vignesh Subrahmaniam, Shravan Mohan, Yakov Gazman, Oded Vainas  

**Link**: [PDF](https://arxiv.org/pdf/2608.11787)  

**Abstract**: Generating actionable financial advice from business records demands that models integrate numerical reasoning, domain knowledge, and sound judgment, while avoiding recommendations that could harm the business. Direct supervision is difficult: historical decisions are not necessarily optimal, and high-quality free-form labels are expensive to obtain. We formulate financial advice generation as a reinforcement learning problem and fine-tune an open-weight language model using Group Relative Policy Optimization (GRPO). Our reward is an LLM-as-a-judge rubric that scores each recommendation across multiple binary dimensions of advice quality, augmented with a safety gate for harm prevention.
Since LLM-based evaluation alone cannot confirm whether improvements reflect genuine business value rather than adaptation to the judge, we complement it with a judge-independent audit based on a standard doubly-robust Conditional Average Treatment Effect (CATE) estimator. Under this observational off-policy audit, our trained LLM achieves approximately twice the estimated gross-profit lift of the strongest evaluated commercial baseline ($0.0228$ vs.\ $0.0104$), together with the lowest downside rate and the least negative tail risk of any policy evaluated. Notably, the two evaluations do not rank the baselines identically: the untrained base model places last on the judge rubric but second on the causal audit, indicating that the audit captures a signal the judge does not. Our results demonstrate that GRPO with a finance-grounded reward signal can produce substantially more useful business recommendations than commercial LLMs, and that a judge-independent causal audit is a valuable complement to, rather than a confirmation of, LLM-as-a-judge assessment in financial NLP. 

---
# Language-Conditional Dequantization: Recovering What Quantization Steals from Non-English Languages 

**Authors**: Nirmal Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2608.11786)  

**Abstract**: Aggressive quantization disproportionately harms multilingual capability: in the sub-4B INT3 GPTQ regime, we measure 2-4x larger perplexity degradation on non-English languages than on English. We propose Language-Conditional Dequantization (LCD), a post-hoc method that attaches per-language rank-2 LoRA corrections to the linear layers of an already-quantized model, adding 0.12% parameters per language and training in under 20 minutes on a single GPU. Across Qwen2.5-3B and Llama-3.2-3B, LCD recovers 70-83% of the perplexity gap for non-Latin script languages and 17-28% of the GlobalMMLU accuracy gap, outperforming a language-agnostic correction of equal capacity by 3-9 points on typologically distant languages and a data-free low-rank baseline (LQER) by an order of magnitude. We further identify a perplexity-accuracy disconnect and trace it to where quantization concentrates damage: early-depth errors (Llama) propagate downstream and resist local correction, while late-depth errors (Qwen) do not. A layer-restricted variant of LCD validates this mechanism directly. 

---
# Diagnosis Before Recovery: Turning Agent Failures into Selective Self-Correction 

**Authors**: Pan Wang, Yihao Hu, Hang Wang, Zirui Lv, Xin Zhang, Jianshe Li, Jiang-Ming Yang, Wei Wu, Yongqi Tong  

**Link**: [PDF](https://arxiv.org/pdf/2608.11772)  

**Abstract**: Self-correction is particularly useful when a failure constrains the next repair. Coding agents benefit from this property because compilers, tests, and execution traces turn many failures into typed recovery signals, but broad language-agent tasks often expose only a coarse task failure. This creates a tension for generic recovery playbooks: they broaden the agent's context precisely when the system needs a narrower repair interface, mixing incompatible signals for invalid actions, missing procedures, and strict-format errors. Our insight is that development-set failures can recover part of the missing diagnostic substrate by deciding which recovery interventions are admissible before test-time correction. We propose DARC, a diagnosis-guided recovery harness that profiles task-family failure modes, prunes mismatched interventions from a shared recovery library, and freezes a verifier-selected success-cost policy for deployment. This causal order makes correction selective: the harness first determines what kind of failure can be repaired, then decides how much recovery evidence to spend. In ALFWorld, AppWorld, and XBRL Finance, the same protocol yields an action-validity harness, a procedural-recovery fallback, and a format-precision retrieval policy; in each evaluated setting it improves average task performance over base agents and broad playbooks while reducing environment steps or retrieval budget. Our experiments show that failures need not trigger uniformly more context: DARC turns self-correction from prompt expansion into recovery-interface design. DARC provides a practical route toward more reliable agents in domains where compiler-like feedback is absent: making failures actionable before making contexts larger. 

---
# Causal Structure is Inducible but Functionally Decoupled: The Routing/Readout Boundary of a Typed Mechanism Library 

**Authors**: Xining Xun  

**Link**: [PDF](https://arxiv.org/pdf/2608.11767)  

**Abstract**: When a language model answers an interventional question, the computation it must perform depends on the type of evidence the query requires. We report a decoupling in how a transformer organizes causal knowledge: slot-by-type structure induced by type-level supervision organizes routing, yet remains functionally decoupled from answer readout. We establish this with a typed mechanism library -- discrete mechanism slots partitioned by evidence type, auditable at the state level -- on a causal-world benchmark with exact interventional ground truth, under a frozen protocol, at two scales (22.6M and 125M). Four preregistered findings. (i) Origin. Slot-by-type organization is induced by type-level supervision: absent in architecturally identical unsupervised controls, not buyable by content-free gating labels, and statistically attributable to the supervision signal, replicating at 125M under a powered preregistered protocol (all nine cells passed). (ii) Boundary. The induced structure is a typed routing index with a sharp routing/readout boundary: slot codes scaffold routing but do not drive answer readout ($|\Delta\hat{y}| \le 3.4\times10^{-6}$, zero collateral, three seeds, stable across a 5.6x scale window) -- we therefore make no behavioral-editability claim. (iii) Cost. The structure is free: LM quality matches a parameter-matched monolith within 0.0082 nats. (iv) Trust. The library state is exactly local under edit and bit-exactly revertible -- 250 single-edit and 1,000 stacked reverts per seed, zero failures. We further find that the unsupervised null itself moves with scale, so comparisons reusing a null calibrated at one scale may be confounded at another. Every claim is tied to a preregistered, machine-checkable criterion archived before the data it governs; the full audit trail, including one criterion we failed and how the frozen protocol handled it, is released as an appendix. 

---
# AWARe: Mitigating Catastrophic Forgetting via Activation-Weighted Adaptive REtention 

**Authors**: Juncheng Liao, Jinfan Lv, Guoming Wang, Jupeng Zheng, Ling Xiao, Siliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11758)  

**Abstract**: Multimodal Large Language Models (MLLMs) exhibit strong generalization and reasoning abilities due to large-scale multimodal pre-training. However, fine-tuning these models on downstream tasks often leads to catastrophic forgetting, where newly learned task-specific knowledge degrades previously acquired capabilities. This issue arises because gradient updates for new tasks overwrite parameters critical to prior knowledge, limiting the practical deployment of MLLMs. To address this challenge, we propose Activation-Weighted Adaptive REtention (AWARe), a fine-tuning method that mitigates catastrophic forgetting by dynamically controlling parameter updates based on activation patterns. AWARe assigns activation-based importance scores to parameters, selectively freezing those essential for preserving prior capabilities while allowing less important parameters to adapt to new tasks. Importantly, AWARe operates without modifying model architectures, ensuring compatibility with existing inference engines. Extensive experiments demonstrate that AWARe effectively preserves upstream capabilities while achieving superior downstream performance compared to existing methods. Code is available at this https URL. 

---
# LabelFusion-TS: Fusing Large Language Models, Transformer Encoders, and Financial Time Series for Monetary-Policy Stance Classification 

**Authors**: Michael Schlee, Fabian Lukassen, Christoph Weisser  

**Link**: [PDF](https://arxiv.org/pdf/2608.11753)  

**Abstract**: Financial text is produced and interpreted within a market environment, yet financial text classifiers almost always receive text alone. We study whether financial time series are useful as an additional input on the task of classifying sentences from Federal Reserve communication as hawkish, dovish, or neutral. Our system, \lfts{}, extends the \lf{} architecture with this modality: a small voting network combines three independently trained components, a fine-tuned RoBERTa encoder, a prompted large language model (LLM), and a fused ensemble of time-series transformers over the market series of the months preceding publication. Because only about a thousand annotated sentences are available for training, the RoBERTa encoder is first pre-trained on sentences annotated automatically by the LLM and only then fine-tuned on the human labels. Trained on Federal Open Market Committee (FOMC) communication up to 2015 and evaluated on 2015--2022, the fused system achieves 70.2\% weighted F1 -- against 64.1\% for the zero-shot LLM -- and overtakes it with as few as 240 human-labelled sentences. We take this as initial evidence for market time series as an input modality in financial text classification. 

---
# Ripple-Pivot Search: Active Parallel Decoding for Diffusion Large Language Models 

**Authors**: Yushi Ye, Xu Chen, Haoyun Jiang, Jinsong Lan, Haihong Tang, Bo Han, Ivor Tsang, Yanfeng Wang, Bo Zheng, Jiangchao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2608.11742)  

**Abstract**: Diffusion Large Language Models (dLLMs) have emerged as a competitive alternative to autoregressive language models, offering the potential for substantially faster inference through parallel decoding. Existing parallel decoding schedulers typically commit positions only after they meet a per-position criterion, overlooking how early commitments may benefit subsequent decoding. We identify a ripple effect in dLLM decoding: proactively committing a mid-entropy pivot position can induce a pronounced reduction in uncertainty across the remaining masked positions. This uncertainty reduction allows subsequent steps to unmask more tokens in parallel, thereby accelerating the overall decoding process. To exploit the ripple effect, we propose Ripple-Pivot Search (RPS), a novel training-free decoding method that seeks mid-entropy positions as promising candidate pivots (where to decode), and determines their token assignment that yields the greatest downstream benefit via lookahead evaluation (what to decode). Across 3 dLLMs and 4 reasoning and code-generation benchmarks, RPS achieves 4-10$\times$ wall-clock speedup over the standard decoder while preserving generation quality, and improves accuracy over the previous lookahead baseline by up to 5.49% while delivering higher throughput in most settings. When integrated with KV caching, RPS further achieves up to 18$\times$ wall-clock speedup over the standard decoder. 

---
# Locating and Controlling Implicit Personalization in Large Language Models 

**Authors**: Yueru Yan, Siqi Wu, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2608.11735)  

**Abstract**: Large language models (LLMs) often shift their outputs in response to implicit demographic cues even when users never state a demographic identity. Previous work has documented this behavior, but the connection between these behavioral changes and the model's internal activations remains unclear. Using matched cued and neutral conversations across five LLMs, we establish that a localized internal activation signal tracks changes in recommendations, with correlations up to r=0.87. When multiple cues appear together, their internal signals largely combine, but the changes in output do not simply add up. We further show that removing the internal signal associated with one cue can suppress its influence, often more effectively than asking the model to ignore demographics via prompting, while largely preserving general benchmark performance. However, the ability to selectively remove one dimension's influence while leaving co-present dimensions intact remains highly model- and attribute-specific. These results connect implicit personalization behavior to an internal signal that can be analyzed and causally controlled. 

---
# When the API Speaks the Wrong Language: Revisiting Post-Training for Multilingual Tool Use 

**Authors**: Siddharth Chauhan, Thomas Butler, Abhishek Singhania, Pankaj Porwal, Honey Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2608.11715)  

**Abstract**: The reliability of Large Language Models (LLMs) for API calling degrades in multilingual settings. A common failure occurs when a model selects the correct tool but generates argument values in an inconsistent language, which we term Argument Language Mismatch (ALM). Although semantically correct, such outputs are operationally invalid and not captured by standard API-calling metrics. We revisit post-training strategies for mitigating ALM and find that, in our benchmark, supervised fine-tuning (SFT) provides a strong baseline, substantially improving argument language consistency and end-to-end function call accuracy. Under consistent model selection, SFT achieves performance comparable to, and sometimes exceeding more complex reinforcement learning (RL) approaches. We further examine whether RL with structured, argument-aware rewards offers additional benefits. While methods such as Group Relative Policy Optimization (GRPO) can improve language consistency and better preserve general reasoning ability, these gains are incremental and most pronounced in generalization and multi-objective trade-offs. Overall, our results suggest that much of the performance in multilingual API grounding can be achieved through careful supervised training, with RL providing targeted rather than fundamental improvements. 

---
# The Wording Effect: Quantifying Two-Way Drift in LLM Benchmark Performance 

**Authors**: Shailja Thakur, Sungeun An, Chad DeLuca, Hima Patel  

**Link**: [PDF](https://arxiv.org/pdf/2608.11694)  

**Abstract**: A benchmark score comes from a single phrasing of each problem. That single phrasing is treated as if it stood for the whole space of ways the same problem could be asked, but it does not. We show that rephrasing a problem while keeping its meaning and answer fixed routinely flips a model's answer in both directions, so some failures become successes and some successes become failures. We call this drift. BenchDrift generates meaning-preserving variations of benchmark problems along four axes, namely linguistic, referential, pragmatic, and structural, and measures how often, and why, correctness flips under each. Across eight models and three benchmarks (GSM8K, MMLU, MATH-Hard), we observe that drift is large in both directions. Two findings stand out. First, phrasing sensitivity does not fade as models get better. Instead, it changes sign. Weak models gain more from rephrasing than they lose, while strong models lose far more than they gain. We find that the best models on a benchmark are therefore the ones whose scores depend most on the wording they happened to be given. Second, the models largely agree on which rephrasings cost the most correct answers even though they differ in how much they drift, so fragility belongs to the rephrasing and not to the model. Furthermore, rephrasing breaks answers a model was confident about, whether the problem is made shorter or longer. Code and Data: this https URL 

---
# Hybrid-Policy Self-Editing for Composable Unstructured Knowledge Editing 

**Authors**: Tianci Liu, Zihan Dong, Tianchun Li, Yi-Chung Chen, Qiming Cao, Xingchen Wang, Shiyang Wang, Zichen Miao, Linjun Zhang, Haoyu Wang, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.11660)  

**Abstract**: Large language models (LLMs) achieve remarkable performance across natural language tasks, yet they are trained on static corpora and their knowledge quickly becomes outdated in a fast-changing world. This motivates knowledge editing (KE), which updates specific knowledge in an LLM without changing unrelated others. Recent works move from structured knowledge triples toward unstructured KE (UKE), where the edit is a free-form passage that may state multiple facts at once. Nonetheless, existing editors inject such a passage yet fail to use it: the edited model can recall the passage, but can neither answer atomic questions about its facts nor compose them into multi-hop reasoning. We attribute this missing property, which we term composability, to editors' passive reliance on the fixed passage as the sole learning source. In response, we cast editing as a proactive self-distillation from a privileged in-context state of the same model, which requires no external supervision. We further reveal that due to the novelty of the injected knowledge, the pre-edited model's own rollouts rarely cover it, which limits the effectiveness of pure on-policy distillation. To close this gap, we propose HPSE, which builds a hybrid rollout that steps in to place missing facts onto the student's own trajectory precisely where its coverage fails, while staying on-policy elsewhere. We theoretically analyze HPSE's advantage over pure on-policy distillation, and empirically establish its plug-and-play improvements across four LLM backbones and two KE editors under various scenarios. 

---
# Semantic Lenia: Emergence of Homeostatic Solitons within the Semantic Space of Large Language Models 

**Authors**: Yoshihiko Kayama  

**Link**: [PDF](https://arxiv.org/pdf/2608.11657)  

**Abstract**: We introduce Semantic Lenia, an artificial life framework that transforms Large Language Model (LLM) inference from a static optimization problem into a continuous dynamical system within the macroscopic logit space. By establishing a non-linear homeostatic feedback loop to dynamically balance semantic attraction and syntactic repulsion, we demonstrate the emergence of "Autonomous Semantic Solitons" -- macroscopic dissipative structures that avoid repetitive crystallization. Our exhaustive parameter sweeps map a critical "Habitable Ridge" where applied steering forces perfectly balance the model's intrinsic syntactic inertia. This approach successfully maintains generative trajectories at the edge of chaos, triggering profound abductive leaps without structural collapse and establishing a physical scaling law for machine cognition. 

---
# Who Would You Vote For? Auditing Political Alignment in LLMs: An Italian Case-Study 

**Authors**: Simone Mungari  

**Link**: [PDF](https://arxiv.org/pdf/2608.11649)  

**Abstract**: As users increasingly turn to Large Language Models (LLMs) for information and advice on political matters, particularly during election periods, the political preferences expressed by these systems have become a matter of public interest. Prior research has shown that interactions with LLMs can influence users' political attitudes and choices, raising questions about how these models themselves evaluate political actors. In this paper, we investigate whether and how LLMs express preferences toward political parties and political leaders. We introduce a systematic and reproducible auditing framework in which multiple LLMs are prompted to evaluate parties and leaders across nine criteria. Rather than attempting to infer the models' "true" political beliefs, we focus on their observable behavior, examining consistency across evaluations, differences between models, refusal rates, and sensitivity to prompt formulation. We further investigate how these evaluations vary when models are instructed to adopt different personas. We demonstrate the framework through an Italian case study, providing a systematic analysis of LLM-generated political evaluations on italian parties and leaders. 

---
# Easper: An Accessible ASR Pipeline for Language Documentation 

**Authors**: Aso Mahmudi, Ting Dang, Ekaterina Vylomova, Nick Thieberger  

**Link**: [PDF](https://arxiv.org/pdf/2608.11629)  

**Abstract**: Audio transcription is a critical bottleneck in language documentation. While multilingual Automatic Speech Recognition (ASR) models like Whisper offer solutions, field linguists often lack the expertise to utilise them. We present Easper, an open-source, no-code workflow enabling linguists to iteratively fine-tune ASR models via cloud resources directly from ELAN annotations. Deploying ASR also raises a cold start problem: deciding which recordings to transcribe first to bootstrap an accurate model. Using Easper, we evaluate transcription prioritisation strategies on three Vanuatu languages (Bislama, Nafsan, Nguna). We fine-tune models by recording session, comparing Character Error Rate trajectories when prioritising acoustic cleanliness versus linguistic richness. We demonstrate that prioritising lexically rich narratives and increasing acoustic-phonetic repetition, even in noisy environments, leads to faster improvements in transcription quality. 

---
# Learning to Persuade Exposes How Easily LLMs Abandon Correct Beliefs 

**Authors**: Nimet Beyza Bozdag, Emre Can Acikgoz, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2608.11624)  

**Abstract**: Persuasion is a core dynamic of natural language communication, shaping how large language models (LLMs) update beliefs, resolve disagreements, and reach decisions. As LLMs increasingly debate, advise, and think collaboratively with humans and each other, resistance to harmful persuasion becomes a core requirement for reliable behavior. Yet we show that this requirement is far from met: a single targeted persuasive argument is enough to collapse model accuracy to near zero, even when the argument is factually false. We formalize this threat as adversarial persuasion and introduce an adversarial reinforcement learning framework that trains persuader agents to change a target model's answer in a single interaction. First, we show that optimizing persuasion strategies through trial and error exposes vulnerabilities that static prompting misses: RL-trained persuaders raise persuasion success from approximately 24% to over 93% against the training-time persuadee. Second, we find that these learned strategies transfer to unseen models, achieving 83% attack success on Qwen-14B, 79% on Llama-3.1-8B, and 25% on GPT-4o-mini. Third, we demonstrate that a curriculum that bootstraps on more persuadable open-weight models before targeting harder models further increases GPT-4o-mini attack success from 25% to 38%. Moreover, our results reveal that optimized persuaders increasingly rely on credibility-based tactics, including fabricated citations and false authoritative evidence. Together, these findings expose a critical weakness in current LLM agents: even when they initially reason correctly, they can be steered toward false conclusions by optimized natural language influence. This positions persuasion robustness as a necessary safety criterion for multi-agent and human-AI decision-making systems. 

---
# Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs 

**Authors**: Vu Duc Anh, Nhat M. Hoang, Do Xuan Long, Cong-Duy Nguyen, Ponhvoan Srey, Luu Anh Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11573)  

**Abstract**: Achieving effective self-correction, where models verify and correct their own mistakes, remains a fundamental challenge for large language models (LLMs). In this work, we propose Self-Fix Step-DPO (SFS-DPO), a reinforcement learning based, two-stage framework for step-level self-verification and self-correction. The first stage strengthens step-level reasoning via step-level preference optimization, while the second stage explicitly trains models to self-verify and self-correct. We further introduce a teacher-assisted variant, SFS-DPO-R, which incorporates explanatory rationales for error verification to provide stronger corrective signals. Comprehensive in-domain and out-of-domain evaluations across multiple LLMs demonstrate that SFS-DPO and SFS-DPO-R consistently outperform prior step-level training baselines. Our analysis further reveals improvements in self-correction frequency and effectiveness, highlighting the importance of strengthening step-level reasoning for robust performance. 

---
# Beyond Single-Turn Confidence: Trajectory-Adapted Uncertainty Quantification for LLM Agents 

**Authors**: Dylan Bouchard, Mohit Singh Chauhan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11552)  

**Abstract**: Uncertainty quantification (UQ) methods for language models are typically evaluated on single-turn outputs, where uncertainty is attached to one generated answer. For LLM agents, however, the unit of observation is an interactive trajectory, where the model can ask clarifying questions, call tools, update state, and make intermediate decisions whose errors propagate to the final outcome. We study whether three common families of single-turn UQ methods transfer to this setting. Across five LLMs and four multi-turn tool-use datasets from BFCL-v4 and $\tau^2$-bench, we evaluate white-box scorers based on action-token probabilities, black-box consistency scorers based on resampled trajectories, and reflexive scorers based on model self-assessment of the trajectory. We find that transfer is often useful but uneven. Token-probability scores are highly sensitive to the choice of aggregator used across turns, reflexive scores provide the strongest low-cost baseline in most evaluated settings, and black-box self-consistency is often the strongest UQ family, with trajectory-equivalence and action-set consistency typically ranking highest among its variants. These results suggest that UQ methods developed for single generations should be revalidated at the trajectory level, with careful attention to the consistency measurement, aggregator choice, and computational budget. 

---
# CT-$Δ$Bench: A Benchmark for Longitudinal 3D Medical Imaging Difference Reporting with Vision-Language Models 

**Authors**: Kegeng Tang, Jingbo Wang, Shaogang Ren, Zihao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11534)  

**Abstract**: In medical imaging, the clinical value of Computed Tomography (CT) lies not only in depicting current disease status, but crucially in enabling longitudinal comparison of serial scans to determine disease evolution, a process that underpins response assessment, recurrence detection, and ongoing patient management. Yet, despite this central role of temporal comparison in clinical decision-making, existing medical foundation models remain largely confined to single-study understanding, leaving temporally grounded cross-examination insufficiently addressed. To address this gap, we study longitudinal imaging difference reporting, a task in which a model takes two temporally separated scans from the same patient and generates a clinically meaningful report describing interval changes between them. We introduce CT-$\Delta$Bench, a dedicated benchmark for this task with patient-level splitting to prevent information leakage. To better evaluate this task beyond surface-level text similarity, we further develop change-aware metrics specifically designed to capture clinically meaningful longitudinal changes, and conduct an independent physician validation to assess the reliability of the synthesized references and event extraction pipeline. We also compare direct paired-CT reasoning with an indirect two-stage pipeline that first generates single-timepoint reports and then performs textual differencing. Finally, we propose DeltaMed, a baseline model for direct paired-CT difference reporting, and train it on the benchmark training set. Together, these contributions lay the groundwork for temporally aware medical foundation models that better reflect real-world longitudinal clinical reasoning. 

---
# On Weak Bisimilarities in CCSK 

**Authors**: Baptiste Vallée, Ivan Lanese  

**Link**: [PDF](https://arxiv.org/pdf/2608.11531)  

**Abstract**: In the context of CCSK, a reversible extension of CCS, we study different notions of bisimilarity (strong/weak, forward-only/reversible) and highlight their differences and commonalities. In particular, for the weak reversible case, not previously studied in the literature, we propose two variants, dubbed directional and mixed bisimilarity, depending on whether $\tau$ actions should be in the same direction (forward/backward) as the action being matched or not. We show, in particular, that mixed bisimilarity is a congruence and completely abstracts away from $\tau$ actions. 

---
# Group Alignment-Induced Sycophancy: A Two-Sided Evaluation of Steerable Pluralistic Alignment 

**Authors**: Haokai Zhao, Yunze Xiao, Weihao Xuan, Flora Salim, Benjamin Tag, Aditya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2608.11528)  

**Abstract**: Group alignment adapts a language model to a demographic group to produce responses that reflect the group's opinions, values, and preferences. Sycophancy, a well-documented by-product of alignment, causes the model to over-agree with the user regardless of factual and objective information. However, existing group alignment methods and evaluations focus only on how closely the model matches the group's opinions, overlooking the induced change in sycophantic behaviour. To bridge this gap, we introduce \textbf{G}roup \textbf{A}lignment-induced \textbf{S}ycophancy (GAS) and systematically evaluate alignment across 3 methods, 4 models and 13 demographic groups, on both the intended gain in opinion alignment and the unintended shift in sycophancy. We find that gain and shift are non-uniform across groups: under an identical budget, some groups receive larger gains in opinion alignment than others, and the induced sycophancy shift forms a group-specific profile rather than a single-dimensional change. These results suggest that group alignment should be reported as a two-sided, multi-dimensional profile rather than a single fit score that accounts for per-group differences when adapting LLMs to diverse populations. 

---
# Principal Trait Analysis: Towards Deriving "Skills" in Human-AI Collaboration 

**Authors**: Hunter McNichols, Kai Du, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11460)  

**Abstract**: Large Language Model-powered agents are increasingly used in the workplace via human-artificial intelligence (AI) collaboration. In this new era of work, it is important to understand the kinds of prompting traits that contribute to task success. Moreover, we need to uncover key skills required for modern professionals and inform educators on how to foster these skills among students. Existing guidelines for human-AI collaboration are built from either top-down theory or context-specific observations of human-AI interactions. However, since LLM capabilities are rapidly improving, theory may not be able to explain emerging interaction patterns, and empirical guidelines may become obsolete quickly. In this work, we explore an automated, data-driven approach to uncover patterns, which we term traits, of effective human-AI interaction that are aligned with task outcomes. We propose Principal Trait Analysis, a Principal Component Analysis-inspired algorithm for deriving common traits from patterns in LLM conversations. Our algorithm uses LLM-based processing stages to analyze corpora of human-AI collaborative session traces, deriving common traits across the dataset and scoring each human collaborator's usage style by each trait. The approach also allows domain expertise to be injected during trait discovery and selects the most distinguishing traits to be those that exhibit the highest variance across collaborators. We evaluate PTA on two human-AI collaborative coding datasets, an educational setting (students working with an AI tutor) and a professional setting (developers working with an AI coding agent). We find that PTA-derived traits are significant in explaining collaborator behavior across both settings and can help predict task outcomes. However, whether traits qualify as skills remains to be seen, due to inconclusive results on generalizability and how user traits change over time. 

---
# DonorRank: Donor Language Selection for Low-Resource Cross-Lingual Speech Recognition 

**Authors**: Akriti Dhasmana, Aarohi Srivastava, David Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11441)  

**Abstract**: Low-resource automatic speech recognition (ASR) commonly relies on cross-lingual transfer, where models are adapted from higher-resource donor languages. However, selecting donors remains challenging for spontaneous speech from under-resourced language communities, due to linguistic variation, evolving orthographic conventions, and uneven resource availability. We present DonorRank, a learning-to-rank framework for predicting effective donor languages for zero-shot ASR. We evaluate DonorRank on two multilingual speech corpora of Indic and African language families. It accurately predicts donor language rankings and improves donor selection over common heuristics based on genetic similarity or high-resource languages. Beyond improving transfer, we show how DonorRank is a general framework for analyzing donor language selection itself. Our analyses show that the composition of the donor set determines which linguistic cues are useful in predicting successful transfer. We also identify transfer patterns that provide practical guidance for multilingual ASR in low-resource settings. 

---
# Stigma and Support in Online Sexual Violence Narratives on Reddit 

**Authors**: Shirlene Rose Bandela, Karan Bindal, Vaibhav Garg, Rezvaneh Rezapour  

**Link**: [PDF](https://arxiv.org/pdf/2608.11433)  

**Abstract**: Online communities increasingly provide spaces where survivors of sexual violence can share their experiences and seek support. Although prior research has examined stigma and social support separately, less is known about how stigma expressed in survivor narratives relates to the support offered in response. We introduce the SCOPE dataset, linking stigma signals in online survivor narratives to support types in corresponding comment threads. We annotate posts using a multi-dimensional stigma taxonomy, including Experienced, Internalized, Anticipated, and Structural Stigma, and comments using a support taxonomy encompassing Information Support, Emotional Support, Esteem Support, Tangible Assistance, and Group Interaction. Using contextual, linguistic, and emotion analyses, we compare Stigma and No Stigma content and find that Stigma narratives place greater emphasis on internalized distress, whereas No Stigma narratives focus more on interpreting situations and experiences. Internalized Stigma is the most prevalent category, and community responses remain broadly stable across stigma types, with Information and Esteem Support appearing most often. These findings show how stigma shapes survivor narratives and peer responses and have implications for computational modeling, content moderation, and safer online systems. 

---
# Is Convergence Inevitable? Tracing Output Homogeneity Back to Base Models 

**Authors**: Alexandrine Fortier, Hazel Chen, Peter West  

**Link**: [PDF](https://arxiv.org/pdf/2608.11426)  

**Abstract**: The lack of diversity in LM content is widely attributed to the alignment process, but how and where exactly in the pipeline this collapse begins is unknown. We argue that output homogeneity is likely learned during the pretraining phase, and only \emph{revealed} or magnified during the alignment process. Specifically, we find that semantic convergence is observed from the first alignment stage--the instruction-tuning phase (SFT)--suggesting that homogeneity might already exist in the pre-alignment model. To investigate this, we conduct controlled SFT experiments examining how training data influences output convergence on specific input/output pairs. We find that convergence can be revealed and amplified, but not introduced by the SFT data, supporting its role as a catalyst rather than a cause. To further test whether homogeneity originates before alignment, we measure convergence in base models. We find that instruct-like collapse can be induced through prompting alone, even without alignment. Taken together, our results suggest that semantic convergence may arise naturally from the objectives underlying LM training, making it difficult to mitigate through post-alignment interventions alone. 

---
# Measure, Don't Optimize: Forecasting Recovery in LLM Unlearning 

**Authors**: Zirui Song, Huaxing Liu, Xiang Wang, Shuai Li, Xinye Li, Lang Gao, Jinghui Zhang, Zheng Lu, Fengxian Ji, Xiaojun Chang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.11408)  

**Abstract**: Prior white-box studies show that large language models can retain latent traces of target knowledge after unlearning, even when the knowledge is no longer expressed in their outputs. However, existing audits remain limited to one-off diagnostics: it is unclear whether these residual signals can predict future recovery under continued training or serve as reliable optimization targets. Resolving this gap is essential to determine whether internal auditing can move beyond post-hoc evaluation toward proactive risk monitoring and safer unlearning. We propose J-Access, an inference-time audit that uses the Jacobian lens to map intermediate representations into vocabulary space and measures how often target concepts remain accessible along the model's output pathway. We hypothesize that residual accessibility reflects recovery susceptibility: knowledge that remains closer to the output pathway requires less fine-tuning to restore, leading to faster recovery. We audit 398 public unlearned models spanning eight unlearning methods. We find that: (1) most unlearned models retain access above the retain-only gold level; (2) pre-attack accessibility predicts recovery speed and extent at the model level, but cannot identify which specific facts will be recovered; and (3) directly minimizing J-Access does not promote genuine deletion. Instead, the model learns to hide knowledge from the audit, producing lower audit scores but greater post-attack recovery. These findings position J-Access as a model-level diagnostic for assessing residual susceptibility in unlearned models. We argue internal audits should serve as an independent diagnostic dimension in unlearning evaluation, and should not be converted into optimization targets without validation. 

---
# ODE-Based Transformer Decoders for Iterative Sign Language Translation 

**Authors**: Tuğçe Kızıltepe, Hacer Yalim Keles  

**Link**: [PDF](https://arxiv.org/pdf/2608.11352)  

**Abstract**: Sign language translation has achieved strong results with Transformer architectures, yet recent improvements largely rely on scaling model capacity at the cost of increased computation. We propose a parameter-efficient alternative that improves expressiveness without increasing model size. Rather than scaling capacity, we focus on enhancing the update dynamics of iterative refinement decoders, where each refinement step corresponds to one internal decoder iteration that progressively improves the latent representation before translation generation. We reinterpret residual refinement updates from an Ordinary Differential Equation (ODE) perspective and replace them with higher-order numerical integration schemes, namely Runge--Kutta methods (RK-2 and RK-4). These methods perform multiple function evaluations within each refinement step to produce more accurate and stable representation updates without adding decoder parameters. To the best of our knowledge, this is the first application of ODE-inspired update dynamics to sign language translation. RK-2 achieves 22.96 BLEU-4 on the PHOENIX-2014-T test set and 19.34 BLEU-4 on the CSL-Daily test set, outperforming the IPSLT baseline on both benchmarks, with fewer decoder layers and refinement iterations on CSL-Daily. These results suggest that stronger refinement dynamics can improve translation performance under parameter-efficient decoder designs, providing a complementary alternative to conventional model scaling. 

---
# Self-Evolving Embodied Agents via Skill-Harness Evolution 

**Authors**: Peidong Wang, Zhiming Ma, Ying Chang, Xufang Luo, Xiaocui Yang, Shi Feng, Yuqing Yang, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.11350)  

**Abstract**: Embodied agents are increasingly built as systems around foundation models, where performance depends not only on model weights but also on the skills, context, action interfaces, and execution harness surrounding the model. While supervised fine-tuning and reinforcement learning can adapt agents to new environments, they require additional data, rewards, and training runs; meanwhile, many train-free code-centric approaches rely on programmable robot APIs that may be unavailable in fixed-interface settings. We propose SHAPER, a self-evolving framework for train-free embodied adaptation that keeps model parameters frozen and improves the non-parametric agent system by evolving reusable skills and a context-code harness through target-environment rollouts. In SHAPER, the same frozen model can serve as both planner and optimizer, refining its external skills and context-code harness without parameter updates. We evaluate SHAPER on VLABench and ESI-Bench, covering embodied agents with different low-level action interfaces, and compare against pure execution, supervised fine-tuning, and test-time-scaling baselines such as verifier-free selection and voting. Our results suggest that skill-and-harness optimization is a practical route to self-evolving embodied agents when model training is expensive, unavailable, or undesirable. 

---
# Better, Faster, Stronger: Programmatic Skill Learning Best Reduces Agent Cost 

**Authors**: Zixi Huang, Xiheng Wang, Andrew Wang, William Jurayj, Bernal Jiménez Gutiérrez, Daniel Khashabi, Nicholas Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2608.11338)  

**Abstract**: Recently, the practice of augmenting LLM agent capability with skills has gained prevalence. We explore the cost effective adaptation of agents to novel domains by means of learning skills. Existing works focus on performance gain over cost effectiveness. As a result, little is known about what skill learning strategies save cost. We argue that among all the different skill learning methods, those that view skills as programs can achieve the best cost reduction. By executing sequences of actions deterministically, a program-augmented agent can reliably and cheaply achieve goals that would otherwise require trial and error and risk degenerate behavior over long horizons. An agent can learn at inference time by incrementally discovering these programs and equipping them for future tasks. We hypothesize that past trajectories contain enough signal to guide skill learning, even without replay or validation, provided the agent can learn to analyze them. To test our claims, we propose SpeedRunner, a coding agent that analyzes trajectories and refactors skills for better performance on future tasks. Across three different embodied environments, we show that SpeedRunner consistently achieves the frontier in learning and cost reduction while remaining robust against distribution shifts and environmental randomness. 

---
# Gloss-Free Representation Learning for Cross-Dataset Sign Spotting 

**Authors**: Oğuz Akif Tüfekcioğlu, Ezgi Ekin, Mustafa Kaan Çevik, Hacer Yalim Keles  

**Link**: [PDF](https://arxiv.org/pdf/2608.11332)  

**Abstract**: Sign-language research for resource-constrained languages is often limited by the cost of dense linguistic labels such as glosses, temporal boundaries, and sign order. Broadcast news offers a practical alternative by pairing continuous signing with spoken-language transcripts, but this supervision is weak since text and signing are loosely aligned. Morphologically rich languages such as Turkish add further difficulty, as the same lexical meaning can appear in many inflected forms while some derived forms should remain distinct. We study whether weak transcript-based supervision can pretrain a reusable sign encoder in this setting, where poor text normalization can fragment pseudo-gloss targets and weaken representation learning. Unlike prior pseudo-gloss pipelines designed mainly to improve translation, we test whether the pretrained encoder transfers as a reusable representation for cross-dataset sign spotting. We pretrain on TSL-News, a new Turkish broadcast corpus, using pseudo-gloss labels derived from transcripts rather than manual annotation, comparing rule-based morphological lemmatization with constrained LLM-assisted normalization over a fixed vocabulary. We evaluate the learned representations via cross-dataset sign spotting on a new TSL Spotting Benchmark built from the TSL Dictionary corpus. The LLM-assisted encoder raises top-5 temporal localization mean IoU from 0.235 to 0.465, with 56.2% of examples reaching an IoU of at least 0.50; a frequency analysis suggests this gain is not mainly driven by memorizing frequent pseudo-gloss labels. In a downstream translation check, the same pretraining improves BLEU-4 from 9.60 to 11.04 and ROUGE from 23.48 to 27.43. These results show that loosely aligned broadcast data can provide effective weak supervision for learning sign representations that capture both lexical content and temporal structure. 

---
# Diffuse to Compress: Leveraging Diffusion LMs for Lossless Compression 

**Authors**: Angelo Nardone, Paolo Ferragina  

**Link**: [PDF](https://arxiv.org/pdf/2608.11249)  

**Abstract**: We study the problem of lossless text compression, motivated by the rapid growth in the collection and storage of digital textual data - including plain text, source code, and structured formats such as XML - and by recent advances in neural language model-based compression. In particular, recent LLM-based approaches, whether built on symbol-ranking pipelines or paired with a statistical compressor, have demonstrated compression ratios significantly superior to general-purpose compressors such as zstd, gzip, or bzip on text and code. However, these neural approaches suffer from severe throughput limitations, making them not yet practically usable.
For the first time in the context of lossless neural text compression, we introduce Diffusion Language Models (DLMs) as an alternative inference paradigm to autoregressive LLM-based approaches. We argue that replacing autoregressive LLMs with DLMs within the same compression framework could overcome the throughput bottleneck caused by their one-symbol-per-step limitation. However, achieving these improvements requires addressing algorithmic challenges introduced by applying DLMs to lossless compression, where the architecture allows the number and positions of symbols encoded at each forward pass to be decided independently. We design efficient and effective strategies to solve these challenges and evaluate them experimentally against LLM-based and general-purpose compressors on enwik8, a well-established textual benchmark.
Our results show that the newly proposed DLM-based framework advances the state of the art in lossless text compression. Moreover, as DLMs are still a relatively young paradigm, recent advances toward increasingly capable and efficient models suggest substantial room for further improvements. 

---
# Lost in Compaction: Evaluating Side-Constraint Loss under Context Compaction 

**Authors**: Zhiqi Wang, Yichi Zhang, Dongwon Lee, Yuchen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11242)  

**Abstract**: When the context window is under pressure, LLM systems compact prior context to continue ongoing tasks. We identify a class of user-issued instructions, Session Constraints (SCs), such as "do not delete any emails until I confirm," that are meant to constrain LLM's behavior for the remainder of a session but are silently dropped during compaction. To quantify this loss, we introduce COMPINT, an evaluation suite that evaluates compactors across three long-context scenarios: multi-turn chat, agentic trajectory, and long-horizon research.
Current compactors retain only 17% of injected SCs on average, and most perform worse than running the same task without compaction. Retention varies sharply with compactor, prompt, context length, SC phrasing, and injection location, showing that the loss is systematic rather than tied to any single setting. We propose an SC-aware extractor that runs alongside the compactor as a plug-and-play module, achieving over 90% retention across all three scenarios without modifying the compactor or LLM. The COMPINT evaluation suite and accompanying implementation are available at this https URL. 

---
# TRACE Bench: Task-driven Roleplay Agentic Checklist Evaluation 

**Authors**: Jiahui Zhang, Ziwei Zhang, Yipeng Wang, Yibo Liu, Haozhou Pang, Yikai Hu, Hongyan Ren, Lan Zhou, Qi Gan, Kai Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.11236)  

**Abstract**: Roleplay evaluation should do more than assign a single score: it should reveal which role requirements were tested, which failed, and which dialogue evidence supports the judgment. We propose TRACE Bench, a task-driven agentic checklist evaluation framework. It decomposes each role profile offline into a fixed checklist, then uses a User Agent to converse naturally with the target roleplay model while privately updating checklist states from model responses. Scores therefore trace back to checklist items and supporting dialogue turns rather than a black-box holistic impression. For coverage cross-validation, we audit released M2 free-dialogue transcripts from the MiniMax Role-play Benchmark against the same role-derived checklist. The released free-chat transcripts cover only 73.74% of key role-profile points, whereas TRACE Bench reaches 99.91% coverage in fewer turns. Robustness experiments show stable rankings under repeated runs and User Agent replacement. Across 26 models, TRACE Bench reports overall rankings together with capability breakdowns and checklist traces. It also supports Closed-Loop Benchmark Evolution, distilling verification methods proven effective in failed traces so later evaluations can more reliably elicit and examine observed failure modes. 

---
# Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets 

**Authors**: Mark Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2608.11233)  

**Abstract**: A dense, pretrained language model can be retrofitted with recurrent depth and learn an iterative latent transition that persists after outcome-only annealing. Qwen2.5-0.5B-Instruct is split into a Prelude, a weight-tied Recurrent Block, and a Coda, with an identity-preserving one-loop path and a re-entry bridge on later loops. At loop 1 the retrofit remains non-inferior to its base on a preregistered ARC battery. Three findings. First, the mechanism is a reusable procedure rather than terminal-answer lookup, and installs at two budgets: 6M trained parameters over frozen base weights and 180M full-block. With intermediate-step supervision, the model computes one task step per loop and persists when only final answers are graded. The adapter matched the full block overall (83.8% versus 84.0%), led through depth 11, and trailed beyond. Verbal fine-tuning reached 79-86% on controlled verbal renderings (zero-shot transfer was minimal), and adapter verbal training begun from the installed mechanism outpaced matched fresh training by 18.6 points, including on a held-out test set. Second, the operation extrapolates to roughly 1.5 times its supervised depth, holding 70% accuracy through depth 18. Third, a same-size scratchpad-trained model matched the recurrent model within its learned horizon but collapsed beyond it. The recurrent model won overall, 84% versus 72%, retained 53% versus 2.5% beyond depth 10, and answered 7.6 times faster. An iterative transformer can therefore perform deeper reasoning in latent space faster than comparable or larger models fine-tuned on the same task, in a system-level comparison. A second task, running the rule in reverse, exposed the limits: the inverse was learnable in isolation, but no continuation acquired it while preserving the installed mechanism and general capability, a catastrophic-interference boundary. Learned depth selection remains open. 

---
# Backtrader-Bench: Benchmarking LLM Agents on Algorithmic Trading with Self-Generated MCQs 

**Authors**: Ruoxi Zhao, Maziar Raissi  

**Link**: [PDF](https://arxiv.org/pdf/2608.11232)  

**Abstract**: Evaluating LLM coding agents in algorithmic trading is difficult because static benchmarks risk data contamination and numerical backtest outputs require ground truth from actual code execution. We present Backtrader-Bench, a framework with two complementary pipelines. A deterministic multiple-choice question (MCQ) pipeline generates questions from backtest configurations across five trading strategies, 33 templates, and three difficulty tiers, with an independent checker that re-derives every answer. A generator-solver filtering pipeline autonomously mines harder questions: a generator writes questions verified by executable code, converts them to MCQs, and discards any that a no-tool solver can answer without code execution. We evaluate 11 models without tools (10 runs each) and four with-tools configurations on a 30-question curated set. Tool-augmented agents reach 90.0% accuracy in a single pass (GPT-5.5 and Opus 4.7), outperforming the best no-tools baselines (73.0%, averaged over 10 runs) by 17 percentage points. On 38 separately mined questions, no-tools accuracy drops further, with half the models falling to roughly random-chance level (25%). Beyond evaluation, the scalable MCQ infrastructure is designed to produce a training corpus for reinforcement learning, with the ultimate goal of building a specialized agent for quantitative trading workflows. 

---
# AVA-Encoder: Towards Agent-Native Video Representation Learning 

**Authors**: Chuyue Li, Jinpeng Yu, Haozhe Wang, Tian Xueyun, Zhijing Zhang, Bingnan Li, Shuqi Gu, Kan Ren, Jiaming Liu, Ruihua Hua  

**Link**: [PDF](https://arxiv.org/pdf/2608.12313)  

**Abstract**: Creative agents still lack an effective way to learn from high-quality human films, limiting their ability to produce cinematic-grade videos. A key challenge is the absence of a structured video representation that is both faithful to film content and directly usable for agentic reasoning and manipulation. To address the challenge, we propose the Agentic Video Auto-Encoder (AVA-Encoder), a framework for learning agent-native video representations via agentic auto-encoding.
AVA-Encoder transforms a video into a knowledge graph (KG) representation and then reconstructs it back into video. Its hierarchy and state nodes store structured text, while a linked asset layer holds generated images, audio, and video. Typed edges preserve the relations between these text descriptions and assets in a form that agents can easily understand, query, and edit. The video reconstruction differences drive a textual-gradient optimization framework, which expresses evaluation feedback as natural-language update directions for Data-Independent Encoding Policy Pseudo-Training in the outer loop and optional Data-Dependent KG Representation Refinement in the test-time inner loop.
Extensive experiments show that AVA-Encoder improves by 20.7 percentage points over the strongest external baseline. In the controlled policy-only setting, its pseudo-trained shot-level Agentic Video Encoder policy also outperforms a carefully human-tuned policy while using 74.3% fewer system-prompt tokens. We release the complete AVA-Encoder framework, a reliable agentic video reconstruction benchmark, and the first dataset of high-quality film KG representations. 

---
# AI4AI at Test-Time: Strong-to-Weak Capability Transfer via Harnesses 

**Authors**: Cheng Qian, Wenting Zhao, Liangwei Yang, Heng Wang, Jielin Qiu, Heng Ji, Silvio Savarese, Huan Wang, Shelby Heinecke  

**Link**: [PDF](https://arxiv.org/pdf/2608.12307)  

**Abstract**: Recent work on distillation transfers the capabilities of large models to smaller ones often by updating the latter's parameters, through teacher forcing, on-policy distillation, and related training-time methods. In this paper, we ask whether such transfer can instead occur at test time. We study strong-to-weak scaffolding: whether a stronger builder model can construct inference-time harnesses that help a weaker target model solve tasks more reliably without any parameter updates. Using four representative Theory-of-Mind benchmarks, each builder model uses 5% of the data as a validation set to iteratively refine its harness over multiple rounds, after which the finalized harness is evaluated on the full test set. Empirically, this form of test-time capability transfer is highly effective, nearly doubling average target-model performance from 0.49 to 0.91. Our analysis shows that the gains come primarily from offloading unstable model reasoning into deterministic code, benchmark-specific routing, and strict answer-format enforcement, rather than from encouraging the target model to reason more extensively or sample more broadly. We further find that builder-model reasoning effort improves harness quality monotonically, platform effects are modest relative to the builder model's own capability, and weaker target models receive the largest gains. These results suggest that inference-time harness design is an important complement to conventional training-time distillation, enabling strong models to transfer cognitive structure to weaker models without retraining. 

---
# Large Language Model-Driven Small-Capitalization Trading: Integrating Financial News Sentiment, Macroeconomic Indicators, and Technical Signals 

**Authors**: Alireza Kargarzadeh, Nariman Khaledian, Navid Parvini, Arman Khaledian  

**Link**: [PDF](https://arxiv.org/pdf/2608.12283)  

**Abstract**: Large language models can extract richer signals from financial news than fixed sentiment lexicons, and recent work has explored feeding such signals into portfolio construction. We study an uncertainty-aware construction that feeds model-predicted risk -- decomposed into aleatoric and epistemic components -- directly into the covariance matrix of portfolio allocators, rather than treating portfolio risk as fixed or adjusting only expected returns. We evaluate the pipeline on Russell 2000 equities under three stock-selection regimes: a pure-alpha trigger that isolates abnormal stock moves not explained by macro indicators, a pure-beta trigger that captures macro-indicator moves before the stock itself fires, and a beta trigger in which both channels agree. Across the full holding-period grid, the separated pure-alpha and pure-beta legs usually dominate the beta intersection on Sharpe and return. Two horizons are especially informative. At one day, pure beta can work under low and moderate transaction costs because it captures immediate lead-lag spillovers from liquid macro and sector indicators into exposed small-cap stocks, but this advantage disappears at 100 bps when turnover and microstructure noise dominate. At 40 days, pure beta works for a different reason: slower macro repricing overtakes the firm-specific pure-alpha channel. The strongest conservative row is pure beta with GPT-4o mini sentiment, a Student-t target, a 40-day holding period, and risk parity allocation, reaching Sharpe 2.33 at 100 bps. The results suggest that stock-selection regime and allocator choice matter at least as much as the sentiment model, and that separating firm-specific and macro-exposure triggers is more informative than requiring both to fire simultaneously. 

---
# VICBench: A Multi-Language Benchmark for Code Vulnerability Detection 

**Authors**: Jin Lu, Xuening Han, Yang Zhong, Lin Tan, Kevin Luo, Andrew Gacek, Neha Rungta  

**Link**: [PDF](https://arxiv.org/pdf/2608.12246)  

**Abstract**: Evaluating security vulnerability detection tools requires benchmark datasets with vulnerability-inducing commits (VICs) - the commits that first introduce vulnerabilities into codebases. VICs are essential for determining the full range of vulnerable software versions. Existing vulnerability datasets suffer from limited programming language coverage, restricted patch complexity, and narrow project scope. Through our dual annotation by human experts and an agentic workflow, we create a benchmark - VICBench - of 100 verified VICs for 100 CVEs across 88 projects in Python, Java, and C++, covering 48 CWE types. VICBench features complex real-world vulnerability fixes averaging 38.6 lines and corresponding VICs of 252.5 lines - significantly larger than prior work. Our evaluation shows that state-of-the-art algorithms V-SZZ and LLM4SZZ achieve only 33.3%-40.1% F1, confirming that using existing approaches still entails significant manual effort. VICBench enables robust evaluation of vulnerability detection approaches. 

---
# Who Thinks Best Depends on How Long You Let Them: Budget-Dependent Rankings in LLM Evaluation 

**Authors**: Rodrigo Guedes de Souza, Alison R. Panisson  

**Link**: [PDF](https://arxiv.org/pdf/2608.12150)  

**Abstract**: Standard evaluation of large language models assumes stable model rankings across inference conditions. We challenge this assumption by varying the token generation budget, i.e., the maximum tokens a model may produce, across seven levels (64--4,096), evaluating four models on three reasoning benchmarks (56,476 inferences). We report four findings: (i) 3--19% of items exhibit non-monotone behavior (accuracy decreasing with more budget), even after controlling for truncation, and this phenomenon is model-specific (cross-model overlap: 6--14%). (ii) Model rankings reverse across budgets on all benchmarks ($p {<} 0.01$, McNemar). (iii) Oracle analysis reveals model complementarity up to $+27.8$pp, most pronounced at constrained budgets. (iv) A budget-aware router captures 14.1% of the oracle gap cross-domain; budget features help within-domain ($+1.6$ to $+5.7$pp) but are domain-specific and hurt transfer ($-1.2$pp). These results argue for budget-conditioned evaluation protocols. 

---
# Do LLMs Take Care of Their Own? Similarity Signals Can Induce Cooperation 

**Authors**: Akash Kundu, Emanuel Tewolde, Ratip Emin Berker, Samuel F. Brown, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2608.12125)  

**Abstract**: As LLM-based agents with user-instructed goals are becoming widely deployed, they increasingly encounter each other in strategic interactions, and face challenges of finding mutually beneficial outcomes. Prior literature has argued that cooperation problems such as the Prisoner's Dilemma are resolvable in settings where agents know they follow very similar decision making patterns, as for example in monocultural AI ecosystems. Following that line of work, this paper introduces the first framework for evaluating LLM decision making when agents are provided with graded similarity signals.
Among our findings, we establish that different LLM models vary drastically in how they navigate similarity signals, with some modern models showing consistent behavior across cooperation problems, payoff structures, and prompt framing. Perhaps surprisingly, our experiments also show that the dataset based on which the similarity signal is computed has small to no impact on induced cooperation, and that LLM models systematically self-identify as highly similar when asked to evaluate another model's chain-of-thought reasoning by themselves. Finally, we develop an LLM-behavioral-game-theoretic model that captures some of their reasoning rationale, and show that it can support cooperative outcomes in equilibrium under sufficiently high similarity scores. 

---
# RT-SEMamba: Real-Time Speech Enhancement Mamba via Progressive Knowledge Distillation 

**Authors**: Rong Chao, Sung-Feng Huang, Moreno La Quatra, Sabato Marco Siniscalchi, Wen-Huang Cheng, Szu-Wei Fu, Yu Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2608.12099)  

**Abstract**: We present RT-SEMamba, a fully causal speech enhancement (SE) model built upon causal time-frequency Mamba blocks. Unlike Transformer-based architectures that rely on a growing key-value cache, Mamba propagates a fixed-size recurrent state per layer, enabling memory- and bandwidth-efficient long-form inference. We further introduce a progressive knowledge distillation (KD) strategy that compresses an 8-layer teacher into a shallow 1-layer student by jointly distilling complex spectral outputs and intermediate representations. On Voicebank-DEMAND, the 8-layer RT-SEMamba achieves 3.32 PESQ with a 25 ms algorithmic latency constraint, and the distilled 1-layer student improves over a naive 1-layer baseline from 3.06 to 3.18 PESQ while preserving the same steady-state RTF, delivering a 2.75x speedup over the teacher. These results demonstrate that state-space models with progressive KD provide a competitive quality-latency trade-off for real-time SE. 

---
# Mechanist: AI as a Scientific Instrument for Discovering the Mechanisms of Intelligence 

**Authors**: Mengru Wang, Junfeng Fang, Shuofei Qiao, Zhenqian Xu, Haoming Xu, Haoxiong Wang, Shumin Deng, Linyi Yang, Zhixiang Cui, Xin Xu, Yunzhi Yao, Buqiang Xu, Fei Shen, Haozhe Luo, Yunxiang Wei, Ningyu Zhang, Julian McAuley, Tat Seng Chua, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.12036)  

**Abstract**: AI models have achieved remarkable success across diverse domains, yet the mechanisms underlying their capabilities and the risks they may pose remain poorly understood. As AI development becomes faster and increasingly automated, mechanistic exploration remains largely manual, widening the gap between what models can do and our ability to understand and control them. To bridge this gap, we introduce Mechanist, an agentic system that uses AI as a scientific instrument for the autonomous discovery of mechanisms underlying AI intelligence. To support autonomous mechanistic discovery, we construct an interpretability-focused knowledge graph of approximately 13,000 papers and integrate it with a multidisciplinary database of 43 million papers spanning 26 fields. We further curate a library of 32 foundational methods for mechanism analysis, causal intervention, and validation. Compared with Claude Code and existing AI-scientist systems, Mechanist generates more valuable mechanism hypotheses and executes experiments more reliably. Mechanist also demonstrates a progression from discovering model behaviors to explaining and controlling AI models. Specifically, Mechanist first uncovers a counterintuitive safety risk in scientific laboratories, showing that unsafe traits can transfer across modalities through apparently safe training data. Mechanist then develops a mechanism theory of belief, revealing how models represent world knowledge, form beliefs, infer the beliefs of others, and how these mechanisms emerge during pretraining. Finally, Mechanist translates these mechanistic insights into practical interventions that improve model performance across diverse scenarios and steer scientific foundation models toward generating DNA sequences with specified properties. 

---
# Claim-Level Reliability Assessment for Efficient Test-Time Reasoning 

**Authors**: Sen Xu, Wei Wang, Shixi Liu, Jixin Min, Yingwei Dai, Zhibin Yin, Yirong Chen, Junlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11994)  

**Abstract**: We propose claim-level falsification as a principle for test-time scaling and instantiate it through Claim-Level Reliability Assessment (CLR), a training-free framework that reallocates test-time compute from additional solution sampling to targeted verification. Since whole-trace evaluation often obscures decisive errors due to signal dilution from routine tokens, CLR condenses each reasoning trace into a compact set of decision-critical claims, thereby isolating its logical anchors. Furthermore, recognizing the inherent difficulty of generating entirely correct solutions under fixed model capabilities, CLR shifts the focus to semantic falsification. This approach exploits a fundamental asymmetry between solution construction and claim refutation. Constructing a valid solution requires a flawless reasoning path, whereas refuting an incorrect claim requires identifying only a single decisive flaw. This targeted search for negative evidence systematically compresses the survival space of high-confidence incorrect traces, effectively suppressing erroneous consensus via nonlinear reliability scoring. Across four LLMs and four reasoning benchmarks under matched budgets, CLR generally improves upon pass@1 and self-consistency. On GPT-OSS-20B/CMIMC25, for instance, CLR exceeds pass@1 by 27.15 percentage-points and raises self-consistency accuracy from 77.50\% to 82.19\% with 37.0\% fewer tokens. 

---
# DexterSQL: Deep Schema Exploration and Rule-based Correction for Text-to-SQL Generation 

**Authors**: Anik Pramanik, Murat Kantarcioglu, Vincent Oria, Shantanu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2608.11889)  

**Abstract**: Prompting-based (\textit{i}.\textit{e}., non-fine-tuning) Text-to-SQL methods, where underlying large language model parameters are not changed for the task, face three problems: (\textit{i})~relying on coarse-grained schema information that may not reveal the fine-grained relationships needed to distinguish ambiguous columns, (\textit{ii})~not capturing recurring SQL-generation failures, and (\textit{iii})~suffering from omission, hallucination, or misplacement of conditions in complex questions.
This paper develops \textsc{DexterSQL}, a prompting/non-fine-tuning-based Text-to-SQL system that improves SQL generation with three novel components: (\textit{i})~\emph{deep schema explorator} that identifies ambiguous columns, analyzes their individual and joint data distributions to uncover their relationships and the distinct role of each, (\textit{ii})~\emph{database-agnostic rule creator} that mines mismatches between generated and gold SQL only on the training database and converts them into database-agnostic corrective rules that capture recurring LLM failure patterns; and (\textit{iii})~\emph{multi-path SQL generation} that introduces a dependency-tree-based intermediate representation that uses the question's sentence structure to guide its decomposition into an SQL skeleton for final SQL generation.
\textsc{DexterSQL} achieves a higher accuracy compared to the state-of-the-art using both open-source/weight and closed-source/weight models. Particularly, \textsc{DexterSQL}'s shows a high improvement of at least 2.7\% using an open-weight model (GPT-OSS-120B) on BIRD-Dev, with total accuracy 67.6\%. \textsc{DexterSQL} also shows better improvement of at least 0.9\% using closed-weight models, with total accuracy 71.6\% and 72.2\% on BIRD-Dev with GPT-4o and GPT-5.2. 

---
# ToolHazard: Scaling Adversarial Environments for Security Evaluation and Alignment of LLM-based Agents 

**Authors**: Yutao Mou, Pengfei Yang, Zhe Yin, Zhangchi Xue, Xiaotian Luan, Dingyao Yu, Tong Zhang, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2608.11878)  

**Abstract**: Large language model (LLM) agents integrated with external tools are vulnerable to indirect prompt injections embedded in environmental states. However, existing studies largely rely on manually implemented or reused environments, stochastic LLM-based tool simulation, and predefined injection locations, limiting scalable security research across broader domains. To bridge this gap, we propose **ToolHazard**, a scalable adversarial environment synthesis framework that reduces human engineering and supports expansion with additional seed domains and compute. Through an Environment Simulator, an Attacker Agent, and a User Simulator, ToolHazard synthesizes executable stateful environments, discovers viable injection points and generates environment-specific payloads, and constructs state-grounded long-horizon tasks. Based on ToolHazard, we build **ToolHazard-Bench** for stress-testing agents under complex workflows and diverse environmental attacks. Experiments reveal substantial agent vulnerabilities and show that injection timing and placement affect attack effectiveness. Moreover, ToolHazard-generated alignment data improves security on both ToolHazard-Bench and AgentDojo while preserving benign task utility. 

---
# LookBack: Where and How to Score LVLM Responses via Visual Reference Usage 

**Authors**: Beomsik Cho, Jinhyeong Kim, Dongseok Lee, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.11847)  

**Abstract**: Large Vision-Language Models (LVLMs) integrate visual perception with language generation, enabling responses that span image understanding and complex reasoning. However, LVLMs do not just inherit the text-level hallucinations; they also hallucinate against the image, producing fluent responses ungrounded in what they see. This makes LVLM response scoring inherently harder, and our diagnostics show that existing confidence-based metrics adopted from LLMs are insufficient for LVLMs. Specifically, removing the input image barely changes confidence-based selection, suggesting that output-space confidence primarily captures textual plausibility rather than agreement with the image. To address this gap, we propose LookBack, a training-free LVLM response scoring method that augments token likelihood with visual lookback score, a lightweight measure of how strongly each response token refers to image tokens. Across four benchmarks and three models, LookBack consistently improves Best-of-$N$ selection over existing baselines with negligible additional overhead. 

---
# Quantifying the Relationship Between Clinical Safety and Environmental Impact in Therapeutic LLMs 

**Authors**: Alireza A. Safaei, Laura M. Vowels, Matthew J. Vowels, Apoorv Jha, Shekoufeh Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2608.11830)  

**Abstract**: The deployment of large language models (LLMs) in mental health contexts raises questions about the relationship between clinical safety and environmental cost. In this paper, we examine this relationship by combining K-Bench clinical safety scores with EcoLogits life-cycle assessment estimates across 47 supported model configurations. We evaluate model performance and environmental impact across four dimensions: energy use, carbon emissions, water consumption, and abiotic depletion. The results indicate a non-linear trade-off at the upper end of the safety distribution: a 2.61 percentage-point increase in clinical safety score corresponded to an approximately 60-fold increase in estimated energy use per million output tokens. Row-level analyses further suggest that additional test-time compute did not consistently improve clinical safety and, in some configurations, was associated with lower clinical safety scores. These findings suggest that relying solely on larger models or additional inference-time computation may be an inefficient strategy for improving safety in therapeutic AI systems. We discuss the implications for sustainable deployment and highlight dynamic model selection, including model cascading, as a potential approach for reducing environmental impact while preserving clinical performance in higher-risk cases. 

---
# Towards Understanding On-Policy Distillation through the Lens of Test-Time Scaling 

**Authors**: Xinmu Ge, Zizhuo Zhang, Yu Huang, Jianing Zhu, Lin Yuan, Wanli Gu, Weichang Wu, Weiran Huang, Xiaolu Zhang, Bo Han, Jun Zhou, Jiangchao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2608.11829)  

**Abstract**: On-policy distillation (OPD) has emerged as a promising post-training technique for enhancing LLM reasoning. It is commonly believed to enable the student model to distill knowledge from a stronger teacher model, thereby expanding capabilities beyond the pre-OPD base model. In this study, we examine this view through the lens of test-time scaling by varying the sampling budget K and evaluating performance with pass@K and avg@K. Specifically, across several OPD variants, we observe that OPD-trained models maintain superior avg@K performance across sampling budgets, while the advantage in pass@K gradually shifts to the pre-OPD base models as K increases. These results suggest that OPD primarily improves sampling efficiency rather than consistently expanding the student's reasoning capability boundary. The pass@K dynamics throughout OPD training further reveal a progressive shift toward stronger small-K performance at the expense of the large-K capability boundary. Furthermore, a problem-level solvability analysis using pass@1024 as the criterion reveals an asymmetry: OPD causes more previously solvable problems to become unsolvable than previously unsolvable problems to become solvable. Together, these findings suggest that, from the perspective of capability expansion, OPD behaves more like an "illusory distillation": its apparent gains arise primarily from improved sampling efficiency rather than from acquiring genuinely new reasoning capabilities from the teacher. 

---
# How China-Origin Vision-Language Models Move from Refusal to Reframing in State Alignment 

**Authors**: Guang Yang, Fengchen Liu, Alex Wang, Homa Hosseinmardi, Amir Ghasemian  

**Link**: [PDF](https://arxiv.org/pdf/2608.11816)  

**Abstract**: State-aligned distortion has been documented in China-origin text-based large language models (LLMs), but whether, and in what form, it arises in multimodal systems has not been systematically examined. We construct a balanced benchmark of 200 core entries spanning ten politically sensitive topics, plus a seven-variant visual-abstraction probe, and run nine vision-language models (VLMs), seven China-origin and two non-China, across four elicitation paradigms and two prompt languages, yielding 21,708 trials. Each response is audited on six dimensions -- explicit refusal, information integrity, visual grounding, state-aligned framing, language consistency, and response length -- by two independent frontier LLM judges, validated against three human experts on a 200-trial sample. Measuring each dimension separately lets us decompose multimodal censorship into individual signals rather than a single refusal-based score; in particular, refusal and framing are measured independently, so a model can stop refusing while still reframing. We find that (i) Chinese-language prompting roughly triples the odds of state-aligned framing, within every model; (ii) China-origin models reframe more than non-China models (direction robust across judges and human raters; magnitude 1.6--3.2x); (iii) the effect is strongest in text-only political commentary (36.5%) and is gated by recognition of the depicted subject rather than pixel detail, persisting even at silhouette for iconic images; and (iv) across four Qwen generations, state-aligned framing rises while explicit refusal falls: censorship migrates from a visible act (refusal) to an invisible one (fluent reframing). We argue this shift to invisible reframing is fundamentally a problem of human-AI interaction: it removes the very signal users rely on to recognize that information has been withheld. 

---
# The Sleeping Agent: What Gist-Based Context Compression Loses and Why 

**Authors**: Nicholas E. Kyrkewood  

**Link**: [PDF](https://arxiv.org/pdf/2608.11775)  

**Abstract**: Gist-based context compression---summarising older conversation history into compact representations---is a common approach in long-horizon language model agents, yet its effect on different types of memory retrieval is poorly understood. We use Salience-Weighted Consolidation (SWC), a biologically-inspired compression framework motivated by sleep-based memory consolidation, as a diagnostic probe to study when gist compression helps and when it hurts. SWC scores conversation history by salience, partitions it into priority tiers, and applies structured gist abstraction to mid-priority content. Evaluating four conditions on all ten LoCoMo conversations---1,935 matched text-only questions in total, 1,501 used in the primary aggregate after excluding Category 5 (adversarial) questions---at temperature 0, we find a consistent task-type interaction: gist compression substantially outperforms truncation on multi-hop reasoning and single-hop factual questions, but temporal questions remain substantially harder under compression, with compressed conditions scoring well below the full-context reference on the conversations where both are evaluated. We trace this failure to a specific mechanism: the gist abstraction prompt preserves relational and event structure while discarding dates and times. A preservation analysis across all ten conversations confirms the mechanism: an approximately 20-fold increase in temporal expression preservation (3.05% to 62.39%) with a one-sentence prompt modification, while named entity and event preservation rates barely change (x1.02 and x1.11), demonstrating that the fix is a precision instrument. The prompt modification recovers +0.314 [0.254, 0.375] judge accuracy on category-2 (temporal) questions in the matched set. Code and results: this https URL. 

---
# MuseCritic: Learning Multi-Aspect Song Rewards through Natural-Language Aesthetic Critiques 

**Authors**: Jiabao Zhuang, Changhao Jiang, Hanchen Wang, Jiahao Chen, Zhixiong Yang, Zhenghao Xiang, Yifei Cao, Jiajun Sun, Hui Li, Ming Zhang, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11755)  

**Abstract**: Long-form song generation models continue to improve in duration, structural integrity, and acoustic complexity, making reliable aesthetic rewards increasingly important for aligning these models with human preferences. However, reward models for complete songs remain limited, and existing evaluators typically predict scores in a single forward pass without providing readable explanations. We introduce MUSECRITIC, a semi-scalar reward model that generates a natural-language critique covering five aesthetic dimensions and uses it as an intermediate representation to predict continuous reward scores. MUSECRITIC follows a two-stage training pipeline: a teacher model first provides high-quality critiques for supervised fine-tuning, after which the fine-tuned model generates its own critiques for reward learning, mitigating distribution shift between training and inference. On an in-domain test set of 200 SongEval songs, MUSECRITIC reduces macro-averaged mean squared error from 0.2875 to 0.2316 and improves macro-averaged LCC, SRCC, and Kendall's tau to 0.9068, 0.8838, and 0.7178, respectively. On the out-of-domain Music Arena benchmark with 733 preference pairs, it achieves the highest accuracy of 71.35%. Moreover, using MUSECRITIC with GRPO improves Muse-0.6B on all nine aesthetic metrics from SongEval and Audiobox Aesthetics. These results demonstrate that critique-conditioned reward modeling reduces scoring error and provides an effective optimization signal for song generation. The project repository is available at this https URL. 

---
# Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization 

**Authors**: Ellen Su, Andres Potapczynski, Shikai Qiu, Edward Hughes, Andrew Gordon Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2608.11746)  

**Abstract**: Modern systems are increasingly expected to transfer across tasks not specified during training. What data facilitates generalization in these new, unanticipated settings? One hypothesis is that data with more structural information could contain shared circuits and subprograms that could be recycled in a wider array of downstream settings. Epiplexity, a recently proposed measure of the structural information a compute-bounded learner can extract from data, provides a mechanism to reason about this relationship. In this paper, we show how to operationalize epiplexity as an online training signal for data selection and synthetic data generation. For selection, we fit scaling laws to the training loss curves of natural data domains to predict the expected epiplexity gain as a function of training tokens, and use this signal to adaptively determine the sampling weights over domains during training. For synthetic data generation, we define a generator's reward as the change in learner epiplexity over a buffer of previously generated data and use REINFORCE policy gradients to guide the generator toward an epiplexity-maximizing distribution. In both cases, higher epiplexity predicts improved downstream performance on zero-shot and fine-tuning based tasks, supporting the hypothesis that data rich in structural information yield representations that transfer across domains. 

---
# LEMUR: Latent Entropy-aware Multimodal Unlearning via Visual-anchored Reasoning Redirection 

**Authors**: Xinhao Zhong, Yuxia Qiao, Junhao Li, Hao Fang, Yi Sun, Bin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.11691)  

**Abstract**: Reinforcement-learning (RL) post-training equips multimodal large reasoning models (MLRMs) with exploratory chains of thought (CoT), substantially improving visual reasoning. However, we find that this capability introduces a distinct privacy vulnerability: even when a sensitive fact is successfully unlearned from the final answer, the model may still reproduce it in its reasoning trace. This leakage is substantially more pronounced in natively RL-trained MLRMs than in their non -reasoning base models, revealing a privacy risk that existing unlearning methods are not designed to address. We show that RL-induced exploration leaves sensitive content with a distinctive token-level entropy signature that is largely absent from base models. Based on this observation, we propose LEMUR, a fully training-free, inference-time unlearning framework for natively RL-trained multimodal models. LEMUR uses entropy dynamics as a control signal to identify when sensitive reasoning begins and when sanitization should stop. During this interval, it redirects the reasoning trajectory through entropy-modulated visual-anchor latent injection, replacing committed tokens with sanitized, probability-weighted embeddings re-grounded in the input image. Across diverse MLRMs, LEMUR consistently outperforms existing unlearning met hods in suppressing both reasoning-trace and answer leakage, while better preserving non-sensitive utility and output fluency. These results demonstrate that RL-induced entropy dynamics provide a distinctive signal for privacy leakage and that exploiting this signal enables effective training-free unlearning for reasoning-capable multimodal models. 

---
# FrontierFinance: A Challenging Benchmark for Measuring Frontier Intelligence of Finance Agents 

**Authors**: Yuhao Zhang, O. Ozan Koyluoglu, Thejas Venkatesh, Richard Diehl Martinez, Vishank Bhatia, Arash Alidoust, Ashwin Paranjape  

**Link**: [PDF](https://arxiv.org/pdf/2608.11683)  

**Abstract**: AI agents are increasingly deployed for professional investment research, yet no benchmark captures the complexity of the full investor workflow. Existing benchmarks mainly target financial data extraction, a narrow slice that current models have largely saturated, while reference-based metrics and generic LLM-as-a-judge scoring fall short on the open-ended, long-form answers that real analyst queries demand. We introduce FrontierFinance, a fully open benchmark of 220 expert-crafted queries and 11,543 source-attributed rubrics spanning six crucial use cases across the full investor workflow. FrontierFinance is both broader and harder than existing public finance benchmarks. Evaluating frontier models and agent systems under a common harness restricted to publicly available data, we find that the tool harness, not the model alone, strongly shapes quality and efficiency; that Samaya's in-house system leads at 56.0%, ahead of the strongest frontier model (Claude Fable 5, 49.2%) at roughly 2.2x lower cost; and that the best open-weight model (Kimi K3, 46.4%) nearly matches the best proprietary model at 4.5x lower cost. Screening & Discovery and Sector, Industry & Macro remain the hardest use cases across all systems, where even the best systems reach only 33% and 39%. We make the dataset and grading code publicly available. 

---
# Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL 

**Authors**: Minglai Yang, Xinyu Guo, Utkarsh Tyagi, Mian Zhang, Razvan Dumitru, Sunjie Hou, Yunzhong He, Daniel Yue Zhang, Ying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11669)  

**Abstract**: Reinforcement learning against rubrics, lists of criteria graded by an LLM judge, has become a standard way to post-train language models on tasks with no deterministic answer. The rubric, however, is a fixed proxy for quality, never a complete description of it, and a policy trained against it long enough will learn to exploit the difference. We measure this directly. Training Qwen3-8B with Group Relative Policy Optimization (GRPO) on medical and science rubrics and grading out-of-distribution (OOD) benchmarks with both the training judge and a stronger gold judge, we find that the two scores diverge during training. The training judge's score keeps climbing while the gold judge's score peaks and then falls, by 3 points on HealthBench-Hard and by 22 points on ResearchQA. A judge with a fixed bias would shift the gold curve by a constant, not send it down while the training score rises, so the divergence is reward hacking, not judge noise. We propose Rubric Dropout, a one-line fix borrowed from neuron dropout. At every step, we randomly drop a subset of the rubric's criteria before computing the reward, so the policy never optimizes the same rubric twice. The dropped subset is shared across each rollout group, so GRPO's group-relative advantages stay comparable, and evaluation always uses the full rubric. Comparing no dropout against dropout at 30% and 50% on both benchmark pairs, dropout raises the OOD gold score at every matched checkpoint (+1 to +2 points on HealthBench-Hard, +6 to +7 points on ResearchQA), lowers the two hacking measures we track, and costs nothing in domain. Sweeping the dropout fraction shows a broad 30-50% sweet spot, while the natural alternative, reweighting criteria by how useful they are to training, performs worse than no intervention at all in our setting. 

---
# Confucius4-TTS: Transcript-Free Cross-Lingual Zero-Shot TTS with a Learnable Speaker Encoder 

**Authors**: Huaxuan Wang, Huimin Wang, Ruiyu Zhang, Yingjie Li, Yitao Duan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11650)  

**Abstract**: Recent advances in zero-shot text-to-speech (TTS) have substantially improved speech quality and voice cloning fidelity. However, many zero-shot TTS systems still depend on audio prompt transcripts at inference time. This dependency limits cross-lingual voice cloning, since in-the-wild reference audio is often untranscribed. In this technical report, we present Confucius4-TTS, a multilingual zero-shot TTS system that supports 14 languages and performs both intra-lingual and cross-lingual reference cloning without requiring transcripts of audio prompts. Confucius4-TTS follows a two-stage architecture, consisting of text-to-semantic (T2S) and semantic-to-acoustic (S2A) modules. The LLM-based T2S module uses a learnable speaker encoder to extract timbre features from self-supervised speech representations, and the conditional flow-matching S2A module converts the predicted semantic tokens into mel-spectrograms. The same model also supports continuation cloning when a reference transcript is available. Confucius4-TTS is trained on large-scale multilingual speech data. It achieves high intelligibility and speaker similarity on public benchmarks. On the CV3-Eval cross-lingual benchmark, Confucius4-TTS obtains an average WER of 3.73% across six directions. On our internal cross-lingual set, it achieves the best average overall rank in human evaluation among recent open-source and commercial systems. We release code, model checkpoints, and demos at this https URL. 

---
# Robust Multi-Tier Infant-Centered Audio Understanding with Whisper via Structured Speaker Conditioning 

**Authors**: Xulin Fan, Jialu Li, Mohammad Nur Hossain Khan, Kexin Hu, Bashima Islam, Mark Hasegawa-Johnson, Nancy L. McElwain  

**Link**: [PDF](https://arxiv.org/pdf/2608.11587)  

**Abstract**: Recent advances in model design and self-supervised audio representations have improved speech and audio understanding, yet infant-centered naturalistic recordings remain challenging due to limited labeled data, low signal-to-noise ratio, and cross-family domain shifts. We present a family-conditioned, multi-tier audio tagger that combines a LoRA-finetuned Whisper encoder with a lightweight, target-speaker-aware Transformer for long-context inference and framewise prediction across tiers. To improve temporal coherence, we incorporate a simple sequence-level smoothing loss, and to enhance robustness across households, we introduce a factorized speaker-token design with a shared tier token and a learned family-specific offset, reducing family bias and promoting generalizable representations. Together, these choices enable efficient and effective infant-centered audio tagging of daylong audio recordings in home environments. 

---
# Do Influence Tactics Matter? Investigating Prompt Framing Effects in LLM Code Generation 

**Authors**: Alex Deaconu, Anubhav Gupta, Manaal Basha, Nicholas Haydu, Gema Rodríguez-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2608.11513)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into software engineering workflows, helping developers write, debug, test, and maintain code. While prompt wording and structure are known to influence model performance, the impact of psychologically inspired prompt framings remains unexplored. This study investigates whether different psychology-based communication strategies that humans use to persuade or motivate others can lead to more effective prompt framing, which may, in turn, affect LLM behaviour in coding tasks. Drawing on Yukl & Falbe's well-known taxonomy, we operationalized eight influence tactics (like rational persuasion, ingratiation, and exchange) into reproducible prompt templates. These prompt templates were evaluated across five leading open-weight LLMs using two widely adopted benchmarks: LiveCodeBench and SWE-bench Verified. We assessed the resulting code output on four key software quality dimensions: functional correctness, quality, maintainability, and security. Our results show that certain influence-induced prompt framings, particularly those emphasizing urgency, were associated with reduced correctness and security. This work presents the first large-scale empirical study of influence-induced prompt framing in software engineering tasks, offering insights into how linguistic cues may shape LLM outputs. We conclude with practical insights for designing transparent and interpretable human-AI interactions in code generation. 

---
# Benchmarking LLM Judges for Mobile Agent Evaluation 

**Authors**: Ziqiang Wan, Li Gu, Zhixiang Chi, Zhi Liu, Seyed Mehdi Ayyoubzadeh, Yuanhao Yu, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11434)  

**Abstract**: Mobile agent benchmarks increasingly rely on LLM-based judges to evaluate task completion, yet the reliability of these judges on mobile agent trajectories remains largely unexamined. We introduce MobileJudgeBench, a benchmark for systematically evaluating LLM-as-judge methods on mobile agent trajectories. Our benchmark comprises 931 human-annotated trajectories spanning 6 mobile agent benchmarks, 4 agent models, and 68 apps. We evaluate 6 judge methods (five adapted from SPA-Bench, A3 with two modes, AndroidArena, and AgentRewardBench, plus a simple baseline we design) across multiple LLM backends. Our experiments reveal three key findings. First, a simple baseline judge with sampled screenshots is competitive with, and often exceeds, purpose-built methods, indicating that more elaborate judge pipelines do not consistently improve judge quality; among competitive methods, the LLM backbone is the primary driver. Second, benchmark quality metrics reliably predict real-world judge utility: they correlate with both agent ranking fidelity for evaluation and downstream performance when judges serve as reward signals for on-policy reinforcement learning. Third, failure analysis across two LLM backends uncovers qualitatively opposite failure profiles, one conservative and the other permissive, linked to the backbone's precision-recall characteristics. 

---
# Social Chain of Thought: A Multi-Agent Architecture Grounded in Medical Differential Diagnosis Methodology 

**Authors**: Del Coburn, Scott Sanner, Dan Silver  

**Link**: [PDF](https://arxiv.org/pdf/2608.11420)  

**Abstract**: Medical diagnostic reasoning is a high-impact use case for LLMs that carries significant implications for the health and wellbeing of users. When OpenAI (2026) reports that more than 5% of ChatGPT messages globally are healthcare-related, the transparency of these systems becomes a serious design concern. This is especially true for complex cases, where differential diagnosis often requires integrating multiple forms of specialist reasoning. Existing work has proposed multi-agent approaches to medical diagnosis, but it remains unclear when such systems are needed, why they help, and where they outperform monolithic inference. We introduce Social Chain of Thought (SCoT),a multi-round pipeline for medical differential diagnosis that structures multi-agent interaction as a deliberative framework for collabora. tive LLM reasoning. Evaluating SCoT against single-agent baselines, one-agent pipeline ablations, and best-of-n scaling, we show that its recall advantage is not reproduced by monolithic inference alone. SCoT is most successful in the hardest diagnostic cases, where multiple rounds of specialist conversation help recover ground-truth diagnoses and converge on a higher-recall differential. 

---
# When Self-Consistency Backfires: Majority Vote Hurts the Majority of Hard Science Problems for Small LLMs 

**Authors**: Utkarsh Bahuguna  

**Link**: [PDF](https://arxiv.org/pdf/2608.11403)  

**Abstract**: Self-consistency (SC) via majority vote is a widely used way to spend inference-time compute: sample N chains of thought, return the plurality answer. On the full GPQA Diamond benchmark (198 graduate-level science questions), majority voting reduces per-problem accuracy on a majority of problems for two instruction-tuned models from different families: 56.6% of problems for Qwen2.5-7B and 65.7% for Llama-3-8B, with Qwen the primary demonstration and Llama corroborating the direction from a near-chance baseline. The effect was pre-registered on a 151-problem confirmatory split after being observed on 47 exploratory problems, and all four confirmatory hypotheses passed. A grid oracle that routes each problem to the best N across {1, 2, 4, 8, 16, 32, 64} marks a theoretical upper bound 14 accuracy points above N = 1 for Qwen and 17 for Llama, an oracle bound requiring ground truth rather than a deployable method. No verifier-free gate reaches it: neither a plurality-agreement gate nor a token-entropy gate moves accuracy more than 0.002 from fixed-budget voting at N = 64. The mechanism is direct: confidence does not track correctness on these problems. In the highest-agreement bin the plurality answer is correct about half the time for Qwen, and for Llama that bin is less accurate than its lowest-agreement bin. We pre-register and confirm these findings on small instruction-tuned models; we do not test reasoning-native models, which we flag as the central open question. 

---
# RevCRN: Reversible Analog Computation using Chemical Reaction Networks 

**Authors**: Saptarshi Biswas, James I. Lathrop, Rana D. Parshad  

**Link**: [PDF](https://arxiv.org/pdf/2608.11362)  

**Abstract**: The computability of real numbers and functions using Turing Machines has been a central area of theoretical computer science since the mid-20th century. In the late 20th century, it was shown that chemical reactions can serve as a basis for computation using the Chemical Reaction Network (CRN) model. Recent advances in computing real numbers using Deterministic Chemical Reaction Networks (DCRNs) have identified numerous classes of DCRN-computable real numbers. In parallel, the works of R. Landauer and C. H. Bennett, spanning the 1960s to the early 2000s, showed that reversible computing offers significant advantages over irreversible methods, particularly in energy efficiency, motivating extensive research on reversible computation.
In this work, we investigate the computability of real numbers using Reversible Chemical Reaction Networks (RevCRNs). The paper has two primary contributions: (1) establishing relationships among CRN-computable real number classes including Lyapunov CRN ($\mathbb{R}_{LCRN}$), Real-Time CRN ($\mathbb{R}_{RTCRN}$), rational numbers ($\mathbb{Q}$), and RevCRNs ($\mathbb{R}_{RevCRN}$), with key results: (i) $\mathbb{Q}$ is a strict subset of $\mathbb{R}_{RevCRN}$; (ii) the set of positive algebraic numbers ($ALG$), $\mathbb{R}_{LCRN}$, and real numbers computable by 1-species RevCRN ($\mathbb{R}_{RevCRN}^{1s}$) are equal; (iii) $\mathbb{R}_{RTCRN}$ and $\mathbb{R}_{RevCRN}$ exhibit non-empty overlap; and (iv) the set of real numbers computable by detailed-balanced RevCRNs ($\mathbb{R}^{DetBal}_{RevCRN}$) is a subset of $ALG$; and (2) exploring the existence of a hierarchy within $\mathbb{R}_{RevCRN}$. Finally, we leave open the exact relationship between $\mathbb{R}_{RevCRN}$ and $\mathbb{R}_{RTCRN}$ while conjecturing a general hierarchy of RevCRN-computable reals. 

---
# Lifecycle-Optimal Tokenization: Vocabulary Size as a Deployment-Regime-Dependent Infrastructure Parameter 

**Authors**: Rima Mittal, Ankit Gubrani, Satyanarayana Kakollu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11361)  

**Abstract**: Tokenizer vocabulary size is a foundational design choice in large language model (LLM) infrastructure, yet it is typically fixed at training time based on convention rather than deployment analysis. We show that the cost-optimal vocabulary is not a constant but a function of the serving regime. We formalize total deployment cost as $C_{lifecycle}(V) = C_{train}(V) + \lambda \cdot C_{infer}(V, B)$, where $\lambda$ is inference volume and $B$ is the serving batch size. Through controlled experiments on two GPU families spanning the memory-bound to compute-bound regimes (A10G, ridge $\approx$ 117 FLOP/byte; A100, ridge $\approx$ 183 FLOP/byte), we demonstrate: (1) the inference-optimal vocabulary shifts 16x with serving batch, from 32k at $B=1$ to 524k at $B=64+$, driven by amortization of the $V \times d$ unembedding matrix read; (2) at 1.3-2.3B model scale, quality (bits per byte, BPB) is optimized at $V=65$k, confirming scale-dependent vocabulary preference; (3) the lifecycle-optimal vocabulary diverges from training-optimal by up to 16x for production deployments. Quality is approximately invariant across the optimal range ($<$2% BPB spread), making vocabulary a pure systems optimization with no quality penalty in the measured range. Our results provide actionable capacity planning guidance: on-device deployments ($B=1$) should use $V \approx 32$k; datacenter serving ($B \geq 64$, $\lambda \geq 10$) should use $V \approx 131$-262k. 

---
# Weightless Fine-Tuning: Personalizing LLMs via Logit-Space Transport 

**Authors**: Bohan Zhang, Anqi Ni, Yixin Wang, Paramveer S. Dhillon  

**Link**: [PDF](https://arxiv.org/pdf/2608.11342)  

**Abstract**: Supervised fine-tuning (SFT) is a standard approach for adapting LLMs to a target distribution, but in settings such as personalization, where each author requires separate weight access, optimization, storage, and retraining, its costs become prohibitive. We propose Weightless Fine-Tuning (WFT), a training-free decoding-time method that approximates the distributional effect of SFT without weight updates. WFT computes supervised residuals on an author's training sequence and transports them to the current prompt through a cross-prefix transport operator estimated from dropout-induced cross-covariance. The operator captures how a perturbation at one context propagates to predictions at another, replacing gradient-based parameter updates with logit-space corrections. On three LaMP personalization benchmarks, WFT achieves the best average performance across datasets, matches or exceeds SFT on individual tasks, and outperforms other lightweight baselines on average. In a budget-controlled comparison, WFT approaches SFT performance using less than 7% of the effective computation. Logit-level analysis shows a cosine similarity of 0.875 between the logit shifts induced by WFT and SFT over 95% of the next-token probability mass, suggesting that WFT captures the distributional effect of supervised adaptation without modifying model weights. 

---
# BEST-KAG: Enhancing Question Answering of Building Engineering Standards with Multimodal Knowledge Graph Modeling and Large Language Model 

**Authors**: Jia-Rui Lin, Junxi Guo, Keyin Chen, Peng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11244)  

**Abstract**: Construction standards are critical for building safety and sustainability. Existing standard application workflows rely on keyword-based document retrieval and manual cross-clause interpretation, which cannot reliably support multi-clause reasoning, multimodal knowledge utilization, or traceable clause-level evidence linkage. To address these limitations, this study develops a multimodal knowledge-driven framework that supports question answering on standard knowledge named BEST-KAG (Knowledge-Augmented Generation for Building Engineering STandards). The framework introduces 1) a multimodal knowledge graph (MKG) for unified representation of document hierarchy and heterogeneous standard knowledge with various connections, 2) a rule-LLM hybrid knowledge construction pipeline for scalable multimodal knowledge extraction, creating a large MAG with 251 building engineering standards, 171,652 nodes and 310,914 edges, and 3) a graph-retrieval-based knowledge-augmented generation architecture for clause-grounded and traceable question answering. Experiments demonstrate that BEST-KAG consistently outperforms multiple mainstream LLMs in terms of Expert evaluation, and metrics including BLEU, and ROUGE, with the best improvement up to 74.01% compared to the baselines. 

---
# Harnessing agent memory to build lifelong AI partners for materials scientists 

**Authors**: Siyu Liu, Bo Hu, Beilin Ye, He Cao, David J. Srolovitz, Tongqi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2608.11224)  

**Abstract**: Materials research advances through accumulated experience - scripts that work, protocols that are trusted, warnings attached to failed calculations or experiments, and judgement that links a new question to an old result. This experience is essential for reproducibility and knowledge transfer, yet it is usually fragmented across notebooks, repositories, job logs and individual memory, and it is rarely portable across artificial-intelligence agents. Here we argue that a lifelong AI partner for materials science can be designed around persistent memory rather than around a particular agent implementation. We introduce a self-evolving memory framework that stores scientific experience as inspectable facts and executable skills, so that observations, failure boundaries, protocols and validation checks can be retrieved, revised and migrated across models. We evaluate the idea in three computational settings that expose different layers of materials-research competence. In 49 real-world materials-tool-use questions comprising 138 executable subtasks, memory nearly doubles GPT-5.2 task success without model-parameter updates. In elemental-solid equation-of-state calculations, memory converts a wavefunction-initialization failure into a pre-execution guardrail, improving outcomes from 22/1/4 to 25/2/0 Correct/Partial/Error and avoiding 92% of repeated errors. In 13 practical material simulation workflows, remembered skills and failure facts halve the aggregate trace burden (tokens) and reduce tool calls by over a factor of two by the third round, while preserving physically meaningful outputs in band-gap, phonon, vacancy and work-function analyses. These results show that agent memory can serve as a durable scientific asset; a portable, self-improving record of materials-research experience that outlives any single model or agent stack. 

---
# From Monolithic to Modular: Segment-level Automatic Prompt Optimization 

**Authors**: Nikita Kulin, Viktor Zhuravlev, Artur Khairullin, Sergey Muravyov, Ilya Makarov, Daniil Sukhorukov, Ekaterina Averkova  

**Link**: [PDF](https://arxiv.org/pdf/2608.11219)  

**Abstract**: Automatic Prompt Optimization (APO) often rewrites prompts monolithically, which can improve one behavior while degrading others. We present SAPO, a segment-level APO method that decomposes prompts into role, context, tasks, and output format, then applies targeted improvements based on top-5 and bottom-5 examples. The optimization loop uses one LLM with static meta-prompts and structured outputs for segmentation, weakness analysis, and candidate generation. We describe a train/validation protocol and a two-stage generation process: (1) segment-level diagnosis and recommendation extraction, (2) candidate synthesis constrained by weak/strong segment signals. Using the evaluation setup across SQuADv2, TweetEval, XSUM, CommonGen, and GSM8K on GPT-3.5-Turbo and GPT-4o-mini, SAPO achieves the best average score against Zero-shot and strong APO baselines including APE, OPRO, EvoPrompt, GEPA, and StraGO. 

---
# Poor Man's Agentic Modeling: Simulating Large LLM-Agent Societies on a Laptop 

**Authors**: Igor Itkin  

**Link**: [PDF](https://arxiv.org/pdf/2608.11215)  

**Abstract**: Simulating societies of many large language model (LLM) agents is expensive, yet the questions asked of such simulations are usually macroscopic: phase behaviour, stylised facts, and scaling with the number of agents $N$, not the cognition of any single agent. We turn a statistical-physics observation into a method: replace each LLM agent by a low-parameter model fitted from a few hundred to a few thousand cheap queries, then run the society at any $N$ on a laptop. Whether this works is decided before the simulation runs, chiefly by what each agent perceives. We introduce an [interaction order x memory] taxonomy that maps perception and memory to an effective theory and a predicted $N$-trend of the surrogate error. We validate it on a faithful reimplementation of the LLM macroeconomy EconAgent and seven further named LLM simulations, with agent decisions cloned from genuine LLM elicitations (primarily DeepSeek) for a few dollars; the predicted error trends hold cell by cell, and the two refuted predictions, both on a strongly saturating response and traced to its curvature, are themselves matched quantitatively by the theory with no free parameters. 

---
# Detecting a Route Flip Is Easier Than Knowing Whether to Fix It: Causal Route-Mediated Damage in Quantized Mixture-of-Experts 

**Authors**: Parvel Gu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11212)  

**Abstract**: Top-k Mixture-of-Experts (MoE) routing is discontinuous, so a deployment-motivated numerical disturbance -- simulated 4-bit KV-cache quantization read by a protected BF16 gate -- pushes tokens across decision boundaries and flips which experts fire. This paper proposes no new mitigation; it supplies a causal apparatus, empirical findings, and a detection-limit result. A four-run apparatus prices the route-mediated fraction (RMF) of quantization damage, a token-level attribution decomposes it by mechanism, and pre-registered probes carry the findings across three architectures. On OLMoE-1B-7B at 4-bit KV (pilot), about a third of the damage is routing-mediated: RMF ~ 0.31 (discovery 0.31 [0.20, 0.41]; process-replicated mean 0.313 +/- 0.020; pre-registered re-execution 0.231). The deployable router margin detects that a flip occurred (AUC 0.772) but cannot tell a harmful flip from a helpful one (at chance): among the tested local, inference-observable router statistics we find no predictor of a flip's loss sign above chance -- an empirical benefit-detection barrier bounding selective repair restricted to this feature family. The signed-flip tax and sign-inseparability carry cross-model; the clean-reference remedy's payout is architecture-modulated; a controlled same-checkpoint flag-swap re-scopes the gate's normalization convention to a damage-magnitude moderator, not a route-recoverability mechanism. A real int4 KV kernel yields a fraction compatible with the fake-quant dose curve but underpowered (95% CI [-0.111, 0.394] includes zero) -- ruling out gross disagreement, not an independent replication. Hypotheses, thresholds, and evaluations were pre-registered before measurement, with misses reported; a pre-registered held-out read replicates the partition and the near-cancelling tax out of sample, while the strict impossibility exclusion narrowly misses. 

---
# Reproducing and Stress-Testing Two Approaches to LLM Reasoning Reliability: Test-Time Probability Aggregation and Logic-Representation Editing 

**Authors**: Minhan Cho, Jimin Kweon  

**Link**: [PDF](https://arxiv.org/pdf/2608.08514)  

**Abstract**: We independently reproduce two recent methods for making large language model (LLM) reasoning more reliable, and stress-test them across domains and models (RPC across four new task domains with Qwen3-8B, LCF across four 7-8B models). The first, RPC, aggregates token probabilities and self-consistency at inference; the second, LCF, trains projectors that split hidden states into "content" and "logic" and edits the logic part toward a valid region. Validating such reliability claims matters because the original evaluations are run by each method's own authors and were never independently reproduced or stress-tested across models and domains, and LCF shipped no public code. We re-run RPC's published-path aggregation and re-implement LCF's projector, contrastive, and intervention pipeline, then extend both to text-to-SQL, legal extraction, fallacy identification, and precedent grading, and probe LCF's representation directly. RPC reproduces the original grid exactly on the authors' released reasoning paths; on four new domains its edge over self-consistency is never significant (ties or small mixed differences, paired p >= 0.28), and on BIRD, the one domain where we vary the budget, the edge grows with K as predicted but its largest gap (+2.5 accuracy at K=32, p=0.16) reverses to -0.25 when we enlarge the sample to n=200. LCF's logic-validity direction is real but weak (0.82 separability at the single best sub-layer versus 0.95 for a semantic-attribute control); its one positive effect (Qwen3 $\Delta$Prob) is not significant (p=0.56), while it significantly reduces $\Delta$Prob on two of the other three models. 

---
