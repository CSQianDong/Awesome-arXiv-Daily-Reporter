# ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models 

**Authors**: Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.24864)  

**Abstract**: Recent advances in reasoning-centric language models have highlighted reinforcement learning (RL) as a promising method for aligning models with verifiable rewards. However, it remains contentious whether RL truly expands a model's reasoning capabilities or merely amplifies high-reward outputs already latent in the base model's distribution, and whether continually scaling up RL compute reliably leads to improved reasoning performance. In this work, we challenge prevailing assumptions by demonstrating that prolonged RL (ProRL) training can uncover novel reasoning strategies that are inaccessible to base models, even under extensive sampling. We introduce ProRL, a novel training methodology that incorporates KL divergence control, reference policy resetting, and a diverse suite of tasks. Our empirical analysis reveals that RL-trained models consistently outperform base models across a wide range of pass@k evaluations, including scenarios where base models fail entirely regardless of the number of attempts. We further show that reasoning boundary improvements correlates strongly with task competence of base model and training duration, suggesting that RL can explore and populate new regions of solution space over time. These findings offer new insights into the conditions under which RL meaningfully expands reasoning boundaries in language models and establish a foundation for future work on long-horizon RL for reasoning. We release model weights to support further research: this https URL 

---
# AlphaOne: Reasoning Models Thinking Slow and Fast at Test Time 

**Authors**: Junyu Zhang, Runpei Dong, Han Wang, Xuying Ning, Haoran Geng, Peihao Li, Xialin He, Yutong Bai, Jitendra Malik, Saurabh Gupta, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24863)  

**Abstract**: This paper presents AlphaOne ($\alpha$1), a universal framework for modulating reasoning progress in large reasoning models (LRMs) at test time. $\alpha$1 first introduces $\alpha$ moment, which represents the scaled thinking phase with a universal parameter $\alpha$. Within this scaled pre-$\alpha$ moment phase, it dynamically schedules slow thinking transitions by modeling the insertion of reasoning transition tokens as a Bernoulli stochastic process. After the $\alpha$ moment, $\alpha$1 deterministically terminates slow thinking with the end-of-thinking token, thereby fostering fast reasoning and efficient answer generation. This approach unifies and generalizes existing monotonic scaling methods by enabling flexible and dense slow-to-fast reasoning modulation. Extensive empirical studies on various challenging benchmarks across mathematical, coding, and scientific domains demonstrate $\alpha$1's superior reasoning capability and efficiency. Project page: this https URL 

---
# MetaFaith: Faithful Natural Language Uncertainty Expression in LLMs 

**Authors**: Gabrielle Kaili-May Liu, Gal Yona, Avi Caciularu, Idan Szpektor, Tim G. J. Rudner, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2505.24858)  

**Abstract**: A critical component in the trustworthiness of LLMs is reliable uncertainty communication, yet LLMs often use assertive language when conveying false claims, leading to over-reliance and eroded trust. We present the first systematic study of $\textit{faithful confidence calibration}$ of LLMs, benchmarking models' ability to use linguistic expressions of uncertainty that $\textit{faithfully reflect}$ their intrinsic uncertainty, across a comprehensive array of models, datasets, and prompting strategies. Our results demonstrate that LLMs largely fail at this task, and that existing interventions are insufficient: standard prompt approaches provide only marginal gains, and existing, factuality-based calibration techniques can even harm faithful calibration. To address this critical gap, we introduce MetaFaith, a novel prompt-based calibration approach inspired by human metacognition. We show that MetaFaith robustly improves faithful calibration across diverse models and task domains, enabling up to 61% improvement in faithfulness and achieving an 83% win rate over original generations as judged by humans. 

---
# Multilinguality Does not Make Sense: Investigating Factors Behind Zero-Shot Transfer in Sense-Aware Tasks 

**Authors**: Roksana Goworek, Haim Dubossarsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.24834)  

**Abstract**: Cross-lingual transfer allows models to perform tasks in languages unseen during training and is often assumed to benefit from increased multilinguality. In this work, we challenge this assumption in the context of two underexplored, sense-aware tasks: polysemy disambiguation and lexical semantic change. Through a large-scale analysis across 28 languages, we show that multilingual training is neither necessary nor inherently beneficial for effective transfer. Instead, we find that confounding factors - such as fine-tuning data composition and evaluation artifacts - better account for the perceived advantages of multilinguality. Our findings call for more rigorous evaluations in multilingual NLP. We release fine-tuned models and benchmarks to support further research, with implications extending to low-resource and typologically diverse languages. 

---
# How much do language models memorize? 

**Authors**: John X. Morris, Chawin Sitawarin, Chuan Guo, Narine Kokhlikyan, G. Edward Suh, Alexander M. Rush, Kamalika Chaudhuri, Saeed Mahloujifar  

**Link**: [PDF](https://arxiv.org/pdf/2505.24832)  

**Abstract**: We propose a new method for estimating how much a model ``knows'' about a datapoint and use it to measure the capacity of modern language models. Prior studies of language model memorization have struggled to disentangle memorization from generalization. We formally separate memorization into two components: \textit{unintended memorization}, the information a model contains about a specific dataset, and \textit{generalization}, the information a model contains about the true data-generation process. When we completely eliminate generalization, we can compute the total memorization, which provides an estimate of model capacity: our measurements estimate that GPT-style models have a capacity of approximately 3.6 bits per parameter. We train language models on datasets of increasing size and observe that models memorize until their capacity fills, at which point ``grokking'' begins, and unintended memorization decreases as models begin to generalize. We train hundreds of transformer language models ranging from $500K$ to $1.5B$ parameters and produce a series of scaling laws relating model capacity and data size to membership inference. 

---
# Improving Reliability and Explainability of Medical Question Answering through Atomic Fact Checking in Retrieval-Augmented LLMs 

**Authors**: Juraj Vladika, Annika Domres, Mai Nguyen, Rebecca Moser, Jana Nano, Felix Busch, Lisa C. Adams, Keno K. Bressem, Denise Bernhardt, Stephanie E. Combs, Kai J. Borm, Florian Matthes, Jan C. Peeken  

**Link**: [PDF](https://arxiv.org/pdf/2505.24830)  

**Abstract**: Large language models (LLMs) exhibit extensive medical knowledge but are prone to hallucinations and inaccurate citations, which pose a challenge to their clinical adoption and regulatory compliance. Current methods, such as Retrieval Augmented Generation, partially address these issues by grounding answers in source documents, but hallucinations and low fact-level explainability persist. In this work, we introduce a novel atomic fact-checking framework designed to enhance the reliability and explainability of LLMs used in medical long-form question answering. This method decomposes LLM-generated responses into discrete, verifiable units called atomic facts, each of which is independently verified against an authoritative knowledge base of medical guidelines. This approach enables targeted correction of errors and direct tracing to source literature, thereby improving the factual accuracy and explainability of medical Q&A. Extensive evaluation using multi-reader assessments by medical experts and an automated open Q&A benchmark demonstrated significant improvements in factual accuracy and explainability. Our framework achieved up to a 40% overall answer improvement and a 50% hallucination detection rate. The ability to trace each atomic fact back to the most relevant chunks from the database provides a granular, transparent explanation of the generated responses, addressing a major gap in current medical AI applications. This work represents a crucial step towards more trustworthy and reliable clinical applications of LLMs, addressing key prerequisites for clinical application and fostering greater confidence in AI-assisted healthcare. 

---
# LegalEval-Q: A New Benchmark for The Quality Evaluation of LLM-Generated Legal Text 

**Authors**: Li yunhan, Wu gengshen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24826)  

**Abstract**: As large language models (LLMs) are increasingly used in legal applications, current evaluation benchmarks tend to focus mainly on factual accuracy while largely neglecting important linguistic quality aspects such as clarity, coherence, and terminology. To address this gap, we propose three steps: First, we develop a regression model to evaluate the quality of legal texts based on clarity, coherence, and terminology. Second, we create a specialized set of legal questions. Third, we analyze 49 LLMs using this evaluation framework.
Our analysis identifies three key findings: First, model quality levels off at 14 billion parameters, with only a marginal improvement of $2.7\%$ noted at 72 billion parameters. Second, engineering choices such as quantization and context length have a negligible impact, as indicated by statistical significance thresholds above 0.016. Third, reasoning models consistently outperform base architectures. A significant outcome of our research is the release of a ranking list and Pareto analysis, which highlight the Qwen3 series as the optimal choice for cost-performance tradeoffs. This work not only establishes standardized evaluation protocols for legal LLMs but also uncovers fundamental limitations in current training data refinement approaches. Code and models are available at: this https URL. 

---
# Guiding Generative Storytelling with Knowledge Graphs 

**Authors**: Zhijun Pan, Antonios Andronis, Eva Hayek, Oscar AP Wilkinson, Ilya Lasy, Annette Parry, Guy Gadney, Tim J. Smith, Mick Grierson  

**Link**: [PDF](https://arxiv.org/pdf/2505.24803)  

**Abstract**: Large Language Models (LLMs) have shown great potential in automated story generation, but challenges remain in maintaining long-form coherence and providing users with intuitive and effective control. Retrieval-Augmented Generation (RAG) has proven effective in reducing hallucinations in text generation; however, the use of structured data to support generative storytelling remains underexplored. This paper investigates how knowledge graphs (KGs) can enhance LLM-based storytelling by improving narrative quality and enabling user-driven modifications. We propose a KG-assisted storytelling pipeline and evaluate its effectiveness through a user study with 15 participants. Participants created their own story prompts, generated stories, and edited knowledge graphs to shape their narratives. Through quantitative and qualitative analysis, our findings demonstrate that knowledge graphs significantly enhance story quality in action-oriented and structured narratives within our system settings. Additionally, editing the knowledge graph increases users' sense of control, making storytelling more engaging, interactive, and playful. 

---
# Drop Dropout on Single-Epoch Language Model Pretraining 

**Authors**: Houjun Liu, John Bauer, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2505.24788)  

**Abstract**: Originally, dropout was seen as a breakthrough regularization technique that reduced overfitting and improved performance in almost all applications of deep learning by reducing overfitting. Yet, single-epoch pretraining tasks common to modern LLMs yield minimal overfitting, leading to dropout not being used for large LLMs. Nevertheless, no thorough empirical investigation has been done on the role of dropout in LM pretraining. Through experiments in single-epoch pretraining of both masked (BERT) and autoregressive (Pythia 160M and 1.4B) LMs with varying levels of dropout, we find that downstream performance in language modeling, morpho-syntax (BLiMP), question answering (SQuAD), and natural-language inference (MNLI) improves when dropout is not applied during pretraining. We additionally find that the recently-introduced "early dropout" also degrades performance over applying no dropout at all. We further investigate the models' editability, and find that models trained without dropout are more successful in gradient-based model editing (MEND) and equivalent in representation-based model editing (ReFT). Therefore, we advocate to drop dropout during single-epoch pretraining. 

---
# Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty? 

**Authors**: Jiayu Liu, Qing Zong, Weiqi Wang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.24778)  

**Abstract**: As large language models (LLMs) are increasingly used in high-stakes domains, accurately assessing their confidence is crucial. Humans typically express confidence through epistemic markers (e.g., "fairly confident") instead of numerical values. However, it remains unclear whether LLMs consistently use these markers to reflect their intrinsic confidence due to the difficulty of quantifying uncertainty associated with various markers. To address this gap, we first define marker confidence as the observed accuracy when a model employs an epistemic marker. We evaluate its stability across multiple question-answering datasets in both in-distribution and out-of-distribution settings for open-source and proprietary LLMs. Our results show that while markers generalize well within the same distribution, their confidence is inconsistent in out-of-distribution scenarios. These findings raise significant concerns about the reliability of epistemic markers for confidence estimation, underscoring the need for improved alignment between marker based confidence and actual model uncertainty. Our code is available at this https URL. 

---
# From Macro to Micro: Probing Dataset Diversity in Language Model Fine-Tuning 

**Authors**: Haoyu Li, Xuhong Li, Yiming Dong, Kun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24768)  

**Abstract**: Dataset diversity plays a pivotal role for the successful training of many machine learning models, particularly in the supervised fine-tuning (SFT) stage of large language model (LLM) development. Despite increasing recognition of its importance, systematic analyses of dataset diversity still remain underexplored. To address this gap, this work presents a systematic taxonomy of existing diversity-control strategies, which primarily focus on the instruction component, operating at either macroscopic (entire instruction semantics) or mesoscopic levels (instruction units), and furthermore introduces a novel analysis of microscopic diversity within the response component, specifically analyzing the statistical distribution of tokens in SFT training samples. In the experimental evaluation, we construct fixed-size datasets (e.g., 10,000 samples each) from a corpus of 117,000 open-source SFT samples, incorporating six distinct diversity-control strategies spanning macro-, meso-, and microscopic levels applied to both instructions and responses. We then fine-tune LLMs on these datasets to assess the six diversity-control strategies. Results reveal that while macroscopic and mesoscopic strategies lead to higher performance with increasing diversity, the microscopic strategy in responses exhibits both a stronger correlation between model performance and the degree of diversity and superior performance with maximum diversity across all strategies. These findings offer actionable insights for constructing high-performance SFT datasets. 

---
# LGAR: Zero-Shot LLM-Guided Neural Ranking for Abstract Screening in Systematic Literature Reviews 

**Authors**: Christian Jaumann, Andreas Wiedholz, Annemarie Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2505.24757)  

**Abstract**: The scientific literature is growing rapidly, making it hard to keep track of the state-of-the-art. Systematic literature reviews (SLRs) aim to identify and evaluate all relevant papers on a topic. After retrieving a set of candidate papers, the abstract screening phase determines initial relevance. To date, abstract screening methods using large language models (LLMs) focus on binary classification settings; existing question answering (QA) based ranking approaches suffer from error propagation. LLMs offer a unique opportunity to evaluate the SLR's inclusion and exclusion criteria, yet, existing benchmarks do not provide them exhaustively. We manually extract these criteria as well as research questions for 57 SLRs, mostly in the medical domain, enabling principled comparisons between approaches. Moreover, we propose LGAR, a zero-shot LLM Guided Abstract Ranker composed of an LLM based graded relevance scorer and a dense re-ranker. Our extensive experiments show that LGAR outperforms existing QA-based methods by 5-10 pp. in mean average precision. Our code and data is publicly available. 

---
# Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation 

**Authors**: Yingchaojie Feng, Yiqun Sun, Yandong Sun, Minfeng Zhu, Qiang Huang, Anthony K. H. Tung, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24754)  

**Abstract**: In this work, we investigate an important task named instruction-following text embedding, which generates dynamic text embeddings that adapt to user instructions, highlighting specific attributes of text. Despite recent advancements, existing approaches suffer from significant computational overhead, as they require re-encoding the entire corpus for each new instruction. To address this challenge, we propose GSTransform, a novel instruction-following text embedding framework based on Guided Space Transformation. Our key observation is that instruction-relevant information is inherently encoded in generic embeddings but remains underutilized. Instead of repeatedly encoding the corpus for each instruction, GSTransform is a lightweight transformation mechanism that adapts pre-computed embeddings in real time to align with user instructions, guided by a small amount of text data with instruction-focused label annotation. We conduct extensive experiments on three instruction-awareness downstream tasks across nine real-world datasets, demonstrating that GSTransform improves instruction-following text embedding quality over state-of-the-art methods while achieving dramatic speedups of 6~300x in real-time processing on large-scale datasets. The source code is available at this https URL. 

---
# Circuit Stability Characterizes Language Model Generalization 

**Authors**: Alan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.24731)  

**Abstract**: Extensively evaluating the capabilities of (large) language models is difficult. Rapid development of state-of-the-art models induce benchmark saturation, while creating more challenging datasets is labor-intensive. Inspired by the recent developments in mechanistic interpretability, we introduce circuit stability as a new way to assess model performance. Circuit stability refers to a model's ability to apply a consistent reasoning process-its circuit-across various inputs. We mathematically formalize circuit stability and circuit equivalence. Then, through three case studies, we empirically show that circuit stability and the lack thereof can characterize and predict different aspects of generalization. Our proposed methods offer a step towards rigorously relating the generality of models to their interpretability. 

---
# Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning 

**Authors**: Shelly Bensal, Umar Jamil, Christopher Bryant, Melisa Russak, Kiran Kamble, Dmytro Mozolevskyi, Muayad Ali, Waseem AlShikh  

**Link**: [PDF](https://arxiv.org/pdf/2505.24726)  

**Abstract**: We explore a method for improving the performance of large language models through self-reflection and reinforcement learning. By incentivizing the model to generate better self-reflections when it answers incorrectly, we demonstrate that a model's ability to solve complex, verifiable tasks can be enhanced even when generating synthetic data is infeasible and only binary feedback is available. Our framework operates in two stages: first, upon failing a given task, the model generates a self-reflective commentary analyzing its previous attempt; second, the model is given another attempt at the task with the self-reflection in context. If the subsequent attempt succeeds, the tokens generated during the self-reflection phase are rewarded. Our experimental results show substantial performance gains across a variety of model architectures, as high as 34.7% improvement at math equation writing and 18.1% improvement at function calling. Notably, smaller fine-tuned models (1.5 billion to 7 billion parameters) outperform models in the same family that are 10 times larger. Our novel paradigm is thus an exciting pathway to more useful and reliable language models that can self-improve on challenging tasks with limited external feedback. 

---
# FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation 

**Authors**: Junyu Luo, Zhizhuo Kou, Liming Yang, Xiao Luo, Jinsheng Huang, Zhiping Xiao, Jingshu Peng, Chengzhong Liu, Jiaming Ji, Xuanzhe Liu, Sirui Han, Ming Zhang, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.24714)  

**Abstract**: Multimodal Large Language Models (MLLMs) have experienced rapid development in recent years. However, in the financial domain, there is a notable lack of effective and specialized multimodal evaluation datasets. To advance the development of MLLMs in the finance domain, we introduce FinMME, encompassing more than 11,000 high-quality financial research samples across 18 financial domains and 6 asset classes, featuring 10 major chart types and 21 subtypes. We ensure data quality through 20 annotators and carefully designed validation mechanisms. Additionally, we develop FinScore, an evaluation system incorporating hallucination penalties and multi-dimensional capability assessment to provide an unbiased evaluation. Extensive experimental results demonstrate that even state-of-the-art models like GPT-4o exhibit unsatisfactory performance on FinMME, highlighting its challenging nature. The benchmark exhibits high robustness with prediction variations under different prompts remaining below 1%, demonstrating superior reliability compared to existing datasets. Our dataset and evaluation protocol are available at this https URL and this https URL. 

---
# Voice Conversion Improves Cross-Domain Robustness for Spoken Arabic Dialect Identification 

**Authors**: Badr M. Abdullah, Matthew Baas, Bernd Möbius, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2505.24713)  

**Abstract**: Arabic dialect identification (ADI) systems are essential for large-scale data collection pipelines that enable the development of inclusive speech technologies for Arabic language varieties. However, the reliability of current ADI systems is limited by poor generalization to out-of-domain speech. In this paper, we present an effective approach based on voice conversion for training ADI models that achieves state-of-the-art performance and significantly improves robustness in cross-domain scenarios. Evaluated on a newly collected real-world test set spanning four different domains, our approach yields consistent improvements of up to +34.1% in accuracy across domains. Furthermore, we present an analysis of our approach and demonstrate that voice conversion helps mitigate the speaker bias in the ADI dataset. We release our robust ADI model and cross-domain evaluation dataset to support the development of inclusive speech technologies for Arabic. 

---
# HESEIA: A community-based dataset for evaluating social biases in large language models, co-designed in real school settings in Latin America 

**Authors**: Guido Ivetta, Marcos J. Gomez, Sofía Martinelli, Pietro Palombini, M. Emilia Echeveste, Nair Carolina Mazzeo, Beatriz Busaniche, Luciana Benotti  

**Link**: [PDF](https://arxiv.org/pdf/2505.24712)  

**Abstract**: Most resources for evaluating social biases in Large Language Models are developed without co-design from the communities affected by these biases, and rarely involve participatory approaches. We introduce HESEIA, a dataset of 46,499 sentences created in a professional development course. The course involved 370 high-school teachers and 5,370 students from 189 Latin-American schools. Unlike existing benchmarks, HESEIA captures intersectional biases across multiple demographic axes and school subjects. It reflects local contexts through the lived experience and pedagogical expertise of educators. Teachers used minimal pairs to create sentences that express stereotypes relevant to their school subjects and communities. We show the dataset diversity in term of demographic axes represented and also in terms of the knowledge areas included. We demonstrate that the dataset contains more stereotypes unrecognized by current LLMs than previous datasets. HESEIA is available to support bias assessments grounded in educational communities. 

---
# Multi-Domain ABSA Conversation Dataset Generation via LLMs for Real-World Evaluation and Model Comparison 

**Authors**: Tejul Pandit, Meet Raval, Dhvani Upadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2505.24701)  

**Abstract**: Aspect-Based Sentiment Analysis (ABSA) offers granular insights into opinions but often suffers from the scarcity of diverse, labeled datasets that reflect real-world conversational nuances. This paper presents an approach for generating synthetic ABSA data using Large Language Models (LLMs) to address this gap. We detail the generation process aimed at producing data with consistent topic and sentiment distributions across multiple domains using GPT-4o. The quality and utility of the generated data were evaluated by assessing the performance of three state-of-the-art LLMs (Gemini 1.5 Pro, Claude 3.5 Sonnet, and DeepSeek-R1) on topic and sentiment classification tasks. Our results demonstrate the effectiveness of the synthetic data, revealing distinct performance trade-offs among the models: DeepSeekR1 showed higher precision, Gemini 1.5 Pro and Claude 3.5 Sonnet exhibited strong recall, and Gemini 1.5 Pro offered significantly faster inference. We conclude that LLM-based synthetic data generation is a viable and flexible method for creating valuable ABSA resources, facilitating research and model evaluation without reliance on limited or inaccessible real-world labeled data. 

---
# Speech-to-Text Translation with Phoneme-Augmented CoT: Enhancing Cross-Lingual Transfer in Low-Resource Scenarios 

**Authors**: Gerard I. Gállego, Oriol Pareras, Martí Cortada Garcia, Lucas Takanori, Javier Hernando  

**Link**: [PDF](https://arxiv.org/pdf/2505.24691)  

**Abstract**: We propose a Speech-to-Text Translation (S2TT) approach that integrates phoneme representations into a Chain-of-Thought (CoT) framework to improve translation in low-resource and zero-resource settings. By introducing phoneme recognition as an intermediate step, we enhance cross-lingual transfer, enabling translation even for languages with no labeled speech data. Our system builds on a multilingual LLM, which we extend to process speech and phonemes. Training follows a curriculum learning strategy that progressively introduces more complex tasks. Experiments on multilingual S2TT benchmarks show that phoneme-augmented CoT improves translation quality in low-resource conditions and enables zero-resource translation, while slightly impacting high-resource performance. Despite this trade-off, our findings demonstrate that phoneme-based CoT is a promising step toward making S2TT more accessible across diverse languages. 

---
# BPE Stays on SCRIPT: Structured Encoding for Robust Multilingual Pretokenization 

**Authors**: Sander Land, Catherine Arnett  

**Link**: [PDF](https://arxiv.org/pdf/2505.24689)  

**Abstract**: Byte Pair Encoding (BPE) tokenizers, widely used in Large Language Models, face challenges in multilingual settings, including penalization of non-Western scripts and the creation of tokens with partial UTF-8 sequences. Pretokenization, often reliant on complex regular expressions, can also introduce fragility and unexpected edge cases. We propose SCRIPT (Script Category Representation in PreTokenization), a novel encoding scheme that bypasses UTF-8 byte conversion by using initial tokens based on Unicode script and category properties. This approach enables a simple, rule-based pretokenization strategy that respects script boundaries, offering a robust alternative to pretokenization strategies based on regular expressions. We also introduce and validate a constrained BPE merging strategy that enforces character integrity, applicable to both SCRIPT-BPE and byte-based BPE. Our experiments demonstrate that SCRIPT-BPE achieves competitive compression while eliminating encoding-based penalties for non-Latin-script languages. 

---
# Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration 

**Authors**: Qinglin Zhu, Runcong Zhao, Hanqi Yan, Yulan He, Yudong Chen, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2505.24688)  

**Abstract**: Large Language Models (LLMs) struggle with complex reasoning due to limited diversity and inefficient search. We propose Soft Reasoning, an embedding-based search framework that optimises the embedding of the first token to guide generation. It combines (1) embedding perturbation for controlled exploration and (2) Bayesian optimisation to refine embeddings via a verifier-guided objective, balancing exploration and exploitation. This approach improves reasoning accuracy and coherence while avoiding reliance on heuristic search. Experiments demonstrate superior correctness with minimal computation, making it a scalable, model-agnostic solution. 

---
# Should I Share this Translation? Evaluating Quality Feedback for User Reliance on Machine Translation 

**Authors**: Dayeon Ki, Kevin Duh, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2505.24683)  

**Abstract**: As people increasingly use AI systems in work and daily life, feedback mechanisms that help them use AI responsibly are urgently needed, particularly in settings where users are not equipped to assess the quality of AI predictions. We study a realistic Machine Translation (MT) scenario where monolingual users decide whether to share an MT output, first without and then with quality feedback. We compare four types of quality feedback: explicit feedback that directly give users an assessment of translation quality using 1) error highlights and 2) LLM explanations, and implicit feedback that helps users compare MT inputs and outputs through 3) backtranslation and 4) question-answer (QA) tables. We find that all feedback types, except error highlights, significantly improve both decision accuracy and appropriate reliance. Notably, implicit feedback, especially QA tables, yields significantly greater gains than explicit feedback in terms of decision accuracy, appropriate reliance, and user perceptions, receiving the highest ratings for helpfulness and trust, and the lowest for mental burden. 

---
# A Simple Linear Patch Revives Layer-Pruned Large Language Models 

**Authors**: Xinrui Chen, Haoli Bai, Tao Yuan, Ruikang Liu, Kang Zhao, Xianzhi Yu, Lu Hou, Tian Guan, Yonghong He, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.24680)  

**Abstract**: Layer pruning has become a popular technique for compressing large language models (LLMs) due to its simplicity. However, existing layer pruning methods often suffer from significant performance drops. We identify that this degradation stems from the mismatch of activation magnitudes across layers and tokens at the pruning interface. To address this, we propose LinearPatch, a simple plug-and-play technique to revive the layer-pruned LLMs. The proposed method adopts Hadamard transformation to suppress massive outliers in particular tokens, and channel-wise scaling to align the activation magnitudes. These operations can be fused into a single matrix, which functions as a patch to bridge the pruning interface with negligible inference overhead. LinearPatch retains up to 94.15% performance of the original model when pruning 5 layers of LLaMA-3-8B on the question answering benchmark, surpassing existing state-of-the-art methods by 4%. In addition, the patch matrix can be further optimized with memory efficient offline knowledge distillation. With only 5K samples, the retained performance of LinearPatch can be further boosted to 95.16% within 30 minutes on a single computing card. 

---
# TRIDENT: Enhancing Large Language Model Safety with Tri-Dimensional Diversified Red-Teaming Data Synthesis 

**Authors**: Xiaorui Wu, Xiaofeng Mao, Fei Li, Xin Zhang, Xuanhong Li, Chong Teng, Donghong Ji, Zhuang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24672)  

**Abstract**: Large Language Models (LLMs) excel in various natural language processing tasks but remain vulnerable to generating harmful content or being exploited for malicious purposes. Although safety alignment datasets have been introduced to mitigate such risks through supervised fine-tuning (SFT), these datasets often lack comprehensive risk coverage. Most existing datasets focus primarily on lexical diversity while neglecting other critical dimensions. To address this limitation, we propose a novel analysis framework to systematically measure the risk coverage of alignment datasets across three essential dimensions: Lexical Diversity, Malicious Intent, and Jailbreak Tactics. We further introduce TRIDENT, an automated pipeline that leverages persona-based, zero-shot LLM generation to produce diverse and comprehensive instructions spanning these dimensions. Each harmful instruction is paired with an ethically aligned response, resulting in two datasets: TRIDENT-Core, comprising 26,311 examples, and TRIDENT-Edge, with 18,773 examples. Fine-tuning Llama 3.1-8B on TRIDENT-Edge demonstrates substantial improvements, achieving an average 14.29% reduction in Harm Score, and a 20% decrease in Attack Success Rate compared to the best-performing baseline model fine-tuned on the WildBreak dataset. 

---
# Multiple LLM Agents Debate for Equitable Cultural Alignment 

**Authors**: Dayeon Ki, Rachel Rudinger, Tianyi Zhou, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2505.24671)  

**Abstract**: Large Language Models (LLMs) need to adapt their predictions to diverse cultural contexts to benefit diverse communities across the world. While previous efforts have focused on single-LLM, single-turn approaches, we propose to exploit the complementary strengths of multiple LLMs to promote cultural adaptability. We introduce a Multi-Agent Debate framework, where two LLM-based agents debate over a cultural scenario and collaboratively reach a final decision. We propose two variants: one where either LLM agents exclusively debate and another where they dynamically choose between self-reflection and debate during their turns. We evaluate these approaches on 7 open-weight LLMs (and 21 LLM combinations) using the NormAd-ETI benchmark for social etiquette norms in 75 countries. Experiments show that debate improves both overall accuracy and cultural group parity over single-LLM baselines. Notably, multi-agent debate enables relatively small LLMs (7-9B) to achieve accuracies comparable to that of a much larger model (27B parameters). 

---
# MSDA: Combining Pseudo-labeling and Self-Supervision for Unsupervised Domain Adaptation in ASR 

**Authors**: Dimitrios Damianos, Georgios Paraskevopoulos, Alexandros Potamianos  

**Link**: [PDF](https://arxiv.org/pdf/2505.24656)  

**Abstract**: In this work, we investigate the Meta PL unsupervised domain adaptation framework for Automatic Speech Recognition (ASR). We introduce a Multi-Stage Domain Adaptation pipeline (MSDA), a sample-efficient, two-stage adaptation approach that integrates self-supervised learning with semi-supervised techniques. MSDA is designed to enhance the robustness and generalization of ASR models, making them more adaptable to diverse conditions. It is particularly effective for low-resource languages like Greek and in weakly supervised scenarios where labeled data is scarce or noisy. Through extensive experiments, we demonstrate that Meta PL can be applied effectively to ASR tasks, achieving state-of-the-art results, significantly outperforming state-of-the-art methods, and providing more robust solutions for unsupervised domain adaptation in ASR. Our ablations highlight the necessity of utilizing a cascading approach when combining self-supervision with self-training. 

---
# PRISM: A Framework for Producing Interpretable Political Bias Embeddings with Political-Aware Cross-Encoder 

**Authors**: Yiqun Sun, Qiang Huang, Anthony K. H. Tung, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24646)  

**Abstract**: Semantic Text Embedding is a fundamental NLP task that encodes textual content into vector representations, where proximity in the embedding space reflects semantic similarity. While existing embedding models excel at capturing general meaning, they often overlook ideological nuances, limiting their effectiveness in tasks that require an understanding of political bias. To address this gap, we introduce PRISM, the first framework designed to Produce inteRpretable polItical biaS eMbeddings. PRISM operates in two key stages: (1) Controversial Topic Bias Indicator Mining, which systematically extracts fine-grained political topics and their corresponding bias indicators from weakly labeled news data, and (2) Cross-Encoder Political Bias Embedding, which assigns structured bias scores to news articles based on their alignment with these indicators. This approach ensures that embeddings are explicitly tied to bias-revealing dimensions, enhancing both interpretability and predictive power. Through extensive experiments on two large-scale datasets, we demonstrate that PRISM outperforms state-of-the-art text embedding models in political bias classification while offering highly interpretable representations that facilitate diversified retrieval and ideological analysis. The source code is available at this https URL. 

---
# Are Optimal Algorithms Still Optimal? Rethinking Sorting in LLM-Based Pairwise Ranking with Batching and Caching 

**Authors**: Juan Wisznia, Cecilia Bolaños, Juan Tollo, Giovanni Marraffini, Agustín Gianolini, Noe Hsueh, Luciano Del Corro  

**Link**: [PDF](https://arxiv.org/pdf/2505.24643)  

**Abstract**: We introduce a novel framework for analyzing sorting algorithms in pairwise ranking prompting (PRP), re-centering the cost model around LLM inferences rather than traditional pairwise comparisons. While classical metrics based on comparison counts have traditionally been used to gauge efficiency, our analysis reveals that expensive LLM inferences overturn these predictions; accordingly, our framework encourages strategies such as batching and caching to mitigate inference costs. We show that algorithms optimal in the classical setting can lose efficiency when LLM inferences dominate the cost under certain optimizations. 

---
# Efficient Text Encoders for Labor Market Analysis 

**Authors**: Jens-Joris Decorte, Jeroen Van Hautte, Chris Develder, Thomas Demeester  

**Link**: [PDF](https://arxiv.org/pdf/2505.24640)  

**Abstract**: Labor market analysis relies on extracting insights from job advertisements, which provide valuable yet unstructured information on job titles and corresponding skill requirements. While state-of-the-art methods for skill extraction achieve strong performance, they depend on large language models (LLMs), which are computationally expensive and slow. In this paper, we propose \textbf{ConTeXT-match}, a novel contrastive learning approach with token-level attention that is well-suited for the extreme multi-label classification task of skill classification. \textbf{ConTeXT-match} significantly improves skill extraction efficiency and performance, achieving state-of-the-art results with a lightweight bi-encoder model. To support robust evaluation, we introduce \textbf{Skill-XL}, a new benchmark with exhaustive, sentence-level skill annotations that explicitly address the redundancy in the large label space. Finally, we present \textbf{JobBERT V2}, an improved job title normalization model that leverages extracted skills to produce high-quality job title representations. Experiments demonstrate that our models are efficient, accurate, and scalable, making them ideal for large-scale, real-time labor market analysis. 

---
# Disentangling Language and Culture for Evaluating Multilingual Large Language Models 

**Authors**: Jiahao Ying, Wei Tang, Yiran Zhao, Yixin Cao, Yu Rong, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24635)  

**Abstract**: This paper introduces a Dual Evaluation Framework to comprehensively assess the multilingual capabilities of LLMs. By decomposing the evaluation along the dimensions of linguistic medium and cultural context, this framework enables a nuanced analysis of LLMs' ability to process questions within both native and cross-cultural contexts cross-lingually. Extensive evaluations are conducted on a wide range of models, revealing a notable "CulturalLinguistic Synergy" phenomenon, where models exhibit better performance when questions are culturally aligned with the language. This phenomenon is further explored through interpretability probing, which shows that a higher proportion of specific neurons are activated in a language's cultural context. This activation proportion could serve as a potential indicator for evaluating multilingual performance during model training. Our findings challenge the prevailing notion that LLMs, primarily trained on English data, perform uniformly across languages and highlight the necessity of culturally and linguistically model evaluations. Our code can be found at https://yingjiahao14. this http URL. 

---
# The Hallucination Dilemma: Factuality-Aware Reinforcement Learning for Large Reasoning Models 

**Authors**: Junyi Li, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2505.24630)  

**Abstract**: Large language models (LLMs) have significantly advanced in reasoning tasks through reinforcement learning (RL) optimization, achieving impressive capabilities across various challenging benchmarks. However, our empirical analysis reveals a critical drawback: reasoning-oriented RL fine-tuning significantly increases the prevalence of hallucinations. We theoretically analyze the RL training dynamics, identifying high-variance gradient, entropy-induced randomness, and susceptibility to spurious local optima as key factors leading to hallucinations. To address this drawback, we propose Factuality-aware Step-wise Policy Optimization (FSPO), an innovative RL fine-tuning algorithm incorporating explicit factuality verification at each reasoning step. FSPO leverages automated verification against given evidence to dynamically adjust token-level advantage values, incentivizing factual correctness throughout the reasoning process. Experiments across mathematical reasoning and hallucination benchmarks using Qwen2.5 and Llama models demonstrate that FSPO effectively reduces hallucinations while enhancing reasoning accuracy, substantially improving both reliability and performance. 

---
# Benchmarking Large Language Models for Cryptanalysis and Mismatched-Generalization 

**Authors**: Utsav Maskey, Chencheng Zhu, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2505.24621)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have transformed natural language understanding and generation, leading to extensive benchmarking across diverse tasks. However, cryptanalysis a critical area for data security and encryption has not yet been thoroughly explored in LLM evaluations. To address this gap, we evaluate cryptanalytic potential of state of the art LLMs on encrypted texts generated using a range of cryptographic algorithms. We introduce a novel benchmark dataset comprising diverse plain texts spanning various domains, lengths, writing styles, and topics paired with their encrypted versions. Using zero-shot and few shot settings, we assess multiple LLMs for decryption accuracy and semantic comprehension across different encryption schemes. Our findings reveal key insights into the strengths and limitations of LLMs in side-channel communication while raising concerns about their susceptibility to jailbreaking attacks. This research highlights the dual-use nature of LLMs in security contexts and contributes to the ongoing discussion on AI safety and security. 

---
# Interpretable phenotyping of Heart Failure patients with Dutch discharge letters 

**Authors**: Vittorio Torri, Machteld J. Boonstra, Marielle C. van de Veerdonk, Deborah N. Kalkman, Alicia Uijl, Francesca Ieva, Ameen Abu-Hanna, Folkert W. Asselbergs, Iacer Calixto  

**Link**: [PDF](https://arxiv.org/pdf/2505.24619)  

**Abstract**: Objective: Heart failure (HF) patients present with diverse phenotypes affecting treatment and prognosis. This study evaluates models for phenotyping HF patients based on left ventricular ejection fraction (LVEF) classes, using structured and unstructured data, assessing performance and interpretability.
Materials and Methods: The study analyzes all HF hospitalizations at both Amsterdam UMC hospitals (AMC and VUmc) from 2015 to 2023 (33,105 hospitalizations, 16,334 patients). Data from AMC were used for model training, and from VUmc for external validation. The dataset was unlabelled and included tabular clinical measurements and discharge letters. Silver labels for LVEF classes were generated by combining diagnosis codes, echocardiography results, and textual mentions. Gold labels were manually annotated for 300 patients for testing. Multiple Transformer-based (black-box) and Aug-Linear (white-box) models were trained and compared with baselines on structured and unstructured data. To evaluate interpretability, two clinicians annotated 20 discharge letters by highlighting information they considered relevant for LVEF classification. These were compared to SHAP and LIME explanations from black-box models and the inherent explanations of Aug-Linear models.
Results: BERT-based and Aug-Linear models, using discharge letters alone, achieved the highest classification results (AUC=0.84 for BERT, 0.81 for Aug-Linear on external validation), outperforming baselines. Aug-Linear explanations aligned more closely with clinicians' explanations than post-hoc explanations on black-box models.
Conclusions: Discharge letters emerged as the most informative source for phenotyping HF patients. Aug-Linear models matched black-box performance while providing clinician-aligned interpretability, supporting their use in transparent clinical decision-making. 

---
# Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX 

**Authors**: Nikita Martynov, Anastasia Mordasheva, Dmitriy Gorbetskiy, Danil Astafurov, Ulyana Isaeva, Elina Basyrova, Sergey Skachkov, Victoria Berestova, Nikolay Ivanov, Valeriia Zanina, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2505.24616)  

**Abstract**: We introduce POLLUX, a comprehensive open-source benchmark designed to evaluate the generative capabilities of large language models (LLMs) in Russian. Our main contribution is a novel evaluation methodology that enhances the interpretability of LLM assessment. For each task type, we define a set of detailed criteria and develop a scoring protocol where models evaluate responses and provide justifications for their ratings. This enables transparent, criteria-driven evaluation beyond traditional resource-consuming, side-by-side human comparisons. POLLUX includes a detailed, fine-grained taxonomy of 35 task types covering diverse generative domains such as code generation, creative writing, and practical assistant use cases, totaling 2,100 manually crafted and professionally authored prompts. Each task is categorized by difficulty (easy/medium/hard), with experts constructing the dataset entirely from scratch. We also release a family of LLM-as-a-Judge (7B and 32B) evaluators trained for nuanced assessment of generative outputs. This approach provides scalable, interpretable evaluation and annotation tools for model development, effectively replacing costly and less precise human judgments. 

---
# Harnessing Large Language Models for Scientific Novelty Detection 

**Authors**: Yan Liu, Zonglin Yang, Soujanya Poria, Thanh-Son Nguyen, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2505.24615)  

**Abstract**: In an era of exponential scientific growth, identifying novel research ideas is crucial and challenging in academia. Despite potential, the lack of an appropriate benchmark dataset hinders the research of novelty detection. More importantly, simply adopting existing NLP technologies, e.g., retrieving and then cross-checking, is not a one-size-fits-all solution due to the gap between textual similarity and idea conception. In this paper, we propose to harness large language models (LLMs) for scientific novelty detection (ND), associated with two new datasets in marketing and NLP domains. To construct the considerate datasets for ND, we propose to extract closure sets of papers based on their relationship, and then summarize their main ideas based on LLMs. To capture idea conception, we propose to train a lightweight retriever by distilling the idea-level knowledge from LLMs to align ideas with similar conception, enabling efficient and accurate idea retrieval for LLM novelty detection. Experiments show our method consistently outperforms others on the proposed benchmark datasets for idea retrieval and ND tasks. Codes and data are available at this https URL. 

---
# When Harry Meets Superman: The Role of The Interlocutor in Persona-Based Dialogue Generation 

**Authors**: Daniela Occhipinti, Marco Guerini, Malvina Nissim  

**Link**: [PDF](https://arxiv.org/pdf/2505.24613)  

**Abstract**: Endowing dialogue agents with persona information has proven to significantly improve the consistency and diversity of their generations. While much focus has been placed on aligning dialogues with provided personas, the adaptation to the interlocutor's profile remains largely underexplored. In this work, we investigate three key aspects: (1) a model's ability to align responses with both the provided persona and the interlocutor's; (2) its robustness when dealing with familiar versus unfamiliar interlocutors and topics, and (3) the impact of additional fine-tuning on specific persona-based dialogues. We evaluate dialogues generated with diverse speaker pairings and topics, framing the evaluation as an author identification task and employing both LLM-as-a-judge and human evaluations. By systematically masking or disclosing information about the interlocutor, we assess its impact on dialogue generation. Results show that access to the interlocutor's persona improves the recognition of the target speaker, while masking it does the opposite. Although models generalise well across topics, they struggle with unfamiliar interlocutors. Finally, we found that in zero-shot settings, LLMs often copy biographical details, facilitating identification but trivialising the task. 

---
# Explainable Depression Detection using Masked Hard Instance Mining 

**Authors**: Patawee Prakrankamanant, Shinji Watanabe, Ekapol Chuangsuwanich  

**Link**: [PDF](https://arxiv.org/pdf/2505.24609)  

**Abstract**: This paper addresses the critical need for improved explainability in text-based depression detection. While offering predictive outcomes, current solutions often overlook the understanding of model predictions which can hinder trust in the system. We propose the use of Masked Hard Instance Mining (MHIM) to enhance the explainability in the depression detection task. MHIM strategically masks attention weights within the model, compelling it to distribute attention across a wider range of salient features. We evaluate MHIM on two datasets representing distinct languages: Thai (Thai-Maywe) and English (DAIC-WOZ). Our results demonstrate that MHIM significantly improves performance in terms of both prediction accuracy and explainability metrics. 

---
# Decoding Knowledge Attribution in Mixture-of-Experts: A Framework of Basic-Refinement Collaboration and Efficiency Analysis 

**Authors**: Junzhuo Li, Bo Wang, Xiuze Zhou, Peijie Jiang, Jia Liu, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24593)  

**Abstract**: The interpretability of Mixture-of-Experts (MoE) models, especially those with heterogeneous designs, remains underexplored. Existing attribution methods for dense models fail to capture dynamic routing-expert interactions in sparse MoE architectures. To address this issue, we propose a cross-level attribution algorithm to analyze sparse MoE architectures (Qwen 1.5-MoE, OLMoE, Mixtral-8x7B) against dense models (Qwen 1.5-7B, Llama-7B, Mixtral-7B). Results show MoE models achieve 37% higher per-layer efficiency via a "mid-activation, late-amplification" pattern: early layers screen experts, while late layers refine knowledge collaboratively. Ablation studies reveal a "basic-refinement" framework--shared experts handle general tasks (entity recognition), while routed experts specialize in domain-specific processing (geographic attributes). Semantic-driven routing is evidenced by strong correlations between attention heads and experts (r=0.68), enabling task-aware coordination. Notably, architectural depth dictates robustness: deep Qwen 1.5-MoE mitigates expert failures (e.g., 43% MRR drop in geographic tasks when blocking top-10 experts) through shared expert redundancy, whereas shallow OLMoE suffers severe degradation (76% drop). Task sensitivity further guides design: core-sensitive tasks (geography) require concentrated expertise, while distributed-tolerant tasks (object attributes) leverage broader participation. These insights advance MoE interpretability, offering principles to balance efficiency, specialization, and robustness. 

---
# GATE: General Arabic Text Embedding for Enhanced Semantic Textual Similarity with Matryoshka Representation Learning and Hybrid Loss Training 

**Authors**: Omer Nacar, Anis Koubaa, Serry Sibaee, Yasser Al-Habashi, Adel Ammar, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2505.24581)  

**Abstract**: Semantic textual similarity (STS) is a critical task in natural language processing (NLP), enabling applications in retrieval, clustering, and understanding semantic relationships between texts. However, research in this area for the Arabic language remains limited due to the lack of high-quality datasets and pre-trained models. This scarcity of resources has restricted the accurate evaluation and advance of semantic similarity in Arabic text. This paper introduces General Arabic Text Embedding (GATE) models that achieve state-of-the-art performance on the Semantic Textual Similarity task within the MTEB benchmark. GATE leverages Matryoshka Representation Learning and a hybrid loss training approach with Arabic triplet datasets for Natural Language Inference, which are essential for enhancing model performance in tasks that demand fine-grained semantic understanding. GATE outperforms larger models, including OpenAI, with a 20-25% performance improvement on STS benchmarks, effectively capturing the unique semantic nuances of Arabic. 

---
# NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization 

**Authors**: Hyuntak Kim, Byung-Hak Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.24575)  

**Abstract**: Summarizing long-form narratives--such as books, movies, and TV scripts--requires capturing intricate plotlines, character interactions, and thematic coherence, a task that remains challenging for existing LLMs. We introduce NexusSum, a multi-agent LLM framework for narrative summarization that processes long-form text through a structured, sequential pipeline--without requiring fine-tuning. Our approach introduces two key innovations: (1) Dialogue-to-Description Transformation: A narrative-specific preprocessing method that standardizes character dialogue and descriptive text into a unified format, improving coherence. (2) Hierarchical Multi-LLM Summarization: A structured summarization pipeline that optimizes chunk processing and controls output length for accurate, high-quality summaries. Our method establishes a new state-of-the-art in narrative summarization, achieving up to a 30.0% improvement in BERTScore (F1) across books, movies, and TV scripts. These results demonstrate the effectiveness of multi-agent LLMs in handling long-form content, offering a scalable approach for structured summarization in diverse storytelling domains. 

---
# Improving Language and Modality Transfer in Translation by Character-level Modeling 

**Authors**: Ioannis Tsiamas, David Dale, Marta R. Costa-jussà  

**Link**: [PDF](https://arxiv.org/pdf/2505.24561)  

**Abstract**: Current translation systems, despite being highly multilingual, cover only 5% of the world's languages. Expanding language coverage to the long-tail of low-resource languages requires data-efficient methods that rely on cross-lingual and cross-modal knowledge transfer. To this end, we propose a character-based approach to improve adaptability to new languages and modalities. Our method leverages SONAR, a multilingual fixed-size embedding space with different modules for encoding and decoding. We use a teacher-student approach with parallel translation data to obtain a character-level encoder. Then, using ASR data, we train a lightweight adapter to connect a massively multilingual CTC ASR model (MMS), to the character-level encoder, potentially enabling speech translation from 1,000+ languages. Experimental results in text translation for 75 languages on FLORES+ demonstrate that our character-based approach can achieve better language transfer than traditional subword-based models, especially outperforming them in low-resource settings, and demonstrating better zero-shot generalizability to unseen languages. Our speech adaptation, maximizing knowledge transfer from the text modality, achieves state-of-the-art results in speech-to-text translation on the FLEURS benchmark on 33 languages, surpassing previous supervised and cascade models, albeit being a zero-shot model with minimal supervision from ASR data. 

---
# Bench4KE: Benchmarking Automated Competency Question Generation 

**Authors**: Anna Sofia Lippolis, Minh Davide Ragagni, Paolo Ciancarini, Andrea Giovanni Nuzzolese, Valentina Presutti  

**Link**: [PDF](https://arxiv.org/pdf/2505.24554)  

**Abstract**: The availability of Large Language Models (LLMs) presents a unique opportunity to reinvigorate research on Knowledge Engineering (KE) automation, a trend already evident in recent efforts developing LLM-based methods and tools for the automatic generation of Competency Questions (CQs). However, the evaluation of these tools lacks standardisation. This undermines the methodological rigour and hinders the replication and comparison of results. To address this gap, we introduce Bench4KE, an extensible API-based benchmarking system for KE automation. Its first release focuses on evaluating tools that generate CQs automatically. CQs are natural language questions used by ontology engineers to define the functional requirements of an ontology. Bench4KE provides a curated gold standard consisting of CQ datasets from four real-world ontology projects. It uses a suite of similarity metrics to assess the quality of the CQs generated. We present a comparative analysis of four recent CQ generation systems, which are based on LLMs, establishing a baseline for future research. Bench4KE is also designed to accommodate additional KE automation tasks, such as SPARQL query generation, ontology testing and drafting. Code and datasets are publicly available under the Apache 2.0 license. 

---
# CREFT: Sequential Multi-Agent LLM for Character Relation Extraction 

**Authors**: Ye Eun Chun, Taeyoon Hwang, Seung-won Hwang, Byung-Hak Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.24553)  

**Abstract**: Understanding complex character relations is crucial for narrative analysis and efficient script evaluation, yet existing extraction methods often fail to handle long-form narratives with nuanced interactions. To address this challenge, we present CREFT, a novel sequential framework leveraging specialized Large Language Model (LLM) agents. First, CREFT builds a base character graph through knowledge distillation, then iteratively refines character composition, relation extraction, role identification, and group assignments. Experiments on a curated Korean drama dataset demonstrate that CREFT significantly outperforms single-agent LLM baselines in both accuracy and completeness. By systematically visualizing character networks, CREFT streamlines narrative comprehension and accelerates script review -- offering substantial benefits to the entertainment, publishing, and educational sectors. 

---
# A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings 

**Authors**: Xiaoang Xu, Shuo Wang, Xu Han, Zhenghao Liu, Huijia Wu, Peipei Li, Zhiyuan Liu, Maosong Sun, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2505.24550)  

**Abstract**: Large Reasoning Models (LRMs) achieve superior performance by extending the thought length. However, a lengthy thinking trajectory leads to reduced efficiency. Most of the existing methods are stuck in the assumption of overthinking and attempt to reason efficiently by compressing the Chain-of-Thought, but this often leads to performance degradation. To address this problem, we introduce A*-Thought, an efficient tree search-based unified framework designed to identify and isolate the most essential thoughts from the extensive reasoning chains produced by these models. It formulates the reasoning process of LRMs as a search tree, where each node represents a reasoning span in the giant reasoning space. By combining the A* search algorithm with a cost function specific to the reasoning path, it can efficiently compress the chain of thought and determine a reasoning path with high information density and low cost. In addition, we also propose a bidirectional importance estimation mechanism, which further refines this search process and enhances its efficiency beyond uniform sampling. Extensive experiments on several advanced math tasks show that A*-Thought effectively balances performance and efficiency over a huge search space. Specifically, A*-Thought can improve the performance of QwQ-32B by 2.39$\times$ with low-budget and reduce the length of the output token by nearly 50% with high-budget. The proposed method is also compatible with several other LRMs, demonstrating its generalization capability. The code can be accessed at: this https URL. 

---
# Cross-Attention Speculative Decoding 

**Authors**: Wei Zhong, Manasa Bharadwaj, Yixiao Wang, Nikhil Verma, Yipeng Ji, Chul Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.24544)  

**Abstract**: Speculative decoding (SD) is a widely adopted approach for accelerating inference in large language models (LLMs), particularly when the draft and target models are well aligned. However, state-of-the-art SD methods typically rely on tightly coupled, self-attention-based Transformer decoders, often augmented with auxiliary pooling or fusion layers. This coupling makes them increasingly complex and harder to generalize across different models. We present Budget EAGLE (Beagle), the first, to our knowledge, cross-attention-based Transformer decoder SD model that achieves performance on par with leading self-attention SD models (EAGLE-v2) while eliminating the need for pooling or auxiliary components, simplifying the architecture, improving training efficiency, and maintaining stable memory usage during training-time simulation. To enable effective training of this novel architecture, we propose Two-Stage Block-Attention Training, a new method that achieves training stability and convergence efficiency in block-level attention scenarios. Extensive experiments across multiple LLMs and datasets show that Beagle achieves competitive inference speedups and higher training efficiency than EAGLE-v2, offering a strong alternative for architectures in speculative decoding. 

---
# Localizing Persona Representations in LLMs 

**Authors**: Celia Cintas, Miriam Rateike, Erik Miehling, Elizabeth Daly, Skyler Speakman  

**Link**: [PDF](https://arxiv.org/pdf/2505.24539)  

**Abstract**: We present a study on how and where personas -- defined by distinct sets of human characteristics, values, and beliefs -- are encoded in the representation space of large language models (LLMs). Using a range of dimension reduction and pattern recognition methods, we first identify the model layers that show the greatest divergence in encoding these representations. We then analyze the activations within a selected layer to examine how specific personas are encoded relative to others, including their shared and distinct embedding spaces. We find that, across multiple pre-trained decoder-only LLMs, the analyzed personas show large differences in representation space only within the final third of the decoder layers. We observe overlapping activations for specific ethical perspectives -- such as moral nihilism and utilitarianism -- suggesting a degree of polysemy. In contrast, political ideologies like conservatism and liberalism appear to be represented in more distinct regions. These findings help to improve our understanding of how LLMs internally represent information and can inform future efforts in refining the modulation of specific human traits in LLM outputs. Warning: This paper includes potentially offensive sample statements. 

---
# Don't Erase, Inform! Detecting and Contextualizing Harmful Language in Cultural Heritage Collections 

**Authors**: Orfeas Menis Mastromichalakis, Jason Liartis, Kristina Rose, Antoine Isaac, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2505.24538)  

**Abstract**: Cultural Heritage (CH) data hold invaluable knowledge, reflecting the history, traditions, and identities of societies, and shaping our understanding of the past and present. However, many CH collections contain outdated or offensive descriptions that reflect historical biases. CH Institutions (CHIs) face significant challenges in curating these data due to the vast scale and complexity of the task. To address this, we develop an AI-powered tool that detects offensive terms in CH metadata and provides contextual insights into their historical background and contemporary perception. We leverage a multilingual vocabulary co-created with marginalized communities, researchers, and CH professionals, along with traditional NLP techniques and Large Language Models (LLMs). Available as a standalone web app and integrated with major CH platforms, the tool has processed over 7.9 million records, contextualizing the contentious terms detected in their metadata. Rather than erasing these terms, our approach seeks to inform, making biases visible and providing actionable insights for creating more inclusive and accessible CH collections. 

---
# DEEPQUESTION: Systematic Generation of Real-World Challenges for Evaluating LLMs Performance 

**Authors**: Ali Khoramfar, Ali Ramezani, Mohammad Mahdi Mohajeri, Mohammad Javad Dousti, Majid Nili Ahmadabadi, Heshaam Faili  

**Link**: [PDF](https://arxiv.org/pdf/2505.24532)  

**Abstract**: LLMs often excel on standard benchmarks but falter on real-world tasks. We introduce DeepQuestion, a scalable automated framework that augments existing datasets based on Bloom's taxonomy and creates novel questions that trace original solution paths to probe evaluative and creative skills. Extensive experiments across ten open-source and proprietary models, covering both general-purpose and reasoning LLMs, reveal substantial performance drops (even up to 70% accuracy loss) on higher-order tasks, underscoring persistent gaps in deep reasoning. Our work highlights the need for cognitively diverse benchmarks to advance LLM progress. DeepQuestion and related datasets will be released upon acceptance of the paper. 

---
# Limited-Resource Adapters Are Regularizers, Not Linguists 

**Authors**: Marcell Fekete, Nathaniel R. Robinson, Ernests Lavrinovics, E. Djeride Jean-Baptiste, Raj Dabre, Johannes Bjerva, Heather Lent  

**Link**: [PDF](https://arxiv.org/pdf/2505.24525)  

**Abstract**: Cross-lingual transfer from related high-resource languages is a well-established strategy to enhance low-resource language technologies. Prior work has shown that adapters show promise for, e.g., improving low-resource machine translation (MT). In this work, we investigate an adapter souping method combined with cross-attention fine-tuning of a pre-trained MT model to leverage language transfer for three low-resource Creole languages, which exhibit relatedness to different language groups across distinct linguistic dimensions. Our approach improves performance substantially over baselines. However, we find that linguistic relatedness -- or even a lack thereof -- does not covary meaningfully with adapter performance. Surprisingly, our cross-attention fine-tuning approach appears equally effective with randomly initialized adapters, implying that the benefit of adapters in this setting lies in parameter regularization, and not in meaningful information transfer. We provide analysis supporting this regularization hypothesis. Our findings underscore the reality that neural language processing involves many success factors, and that not all neural methods leverage linguistic knowledge in intuitive ways. 

---
# Stress-testing Machine Generated Text Detection: Shifting Language Models Writing Style to Fool Detectors 

**Authors**: Andrea Pedrotti, Michele Papucci, Cristiano Ciaccio, Alessio Miaschi, Giovanni Puccetti, Felice Dell'Orletta, Andrea Esuli  

**Link**: [PDF](https://arxiv.org/pdf/2505.24523)  

**Abstract**: Recent advancements in Generative AI and Large Language Models (LLMs) have enabled the creation of highly realistic synthetic content, raising concerns about the potential for malicious use, such as misinformation and manipulation. Moreover, detecting Machine-Generated Text (MGT) remains challenging due to the lack of robust benchmarks that assess generalization to real-world scenarios. In this work, we present a pipeline to test the resilience of state-of-the-art MGT detectors (e.g., Mage, Radar, LLM-DetectAIve) to linguistically informed adversarial attacks. To challenge the detectors, we fine-tune language models using Direct Preference Optimization (DPO) to shift the MGT style toward human-written text (HWT). This exploits the detectors' reliance on stylistic clues, making new generations more challenging to detect. Additionally, we analyze the linguistic shifts induced by the alignment and which features are used by detectors to detect MGT texts. Our results show that detectors can be easily fooled with relatively few examples, resulting in a significant drop in detection performance. This highlights the importance of improving detection methods and making them robust to unseen in-domain texts. 

---
# TimeHC-RL: Temporal-aware Hierarchical Cognitive Reinforcement Learning for Enhancing LLMs' Social Intelligence 

**Authors**: Guiyang Hou, Xing Gao, Yuchuan Wu, Xiang Huang, Wenqi Zhang, Zhe Zheng, Yongliang Shen, Jialu Du, Fei Huang, Yongbin Li, Weiming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24500)  

**Abstract**: Recently, Large Language Models (LLMs) have made significant progress in IQ-related domains that require careful thinking, such as mathematics and coding. However, enhancing LLMs' cognitive development in social domains, particularly from a post-training perspective, remains underexplored. Recognizing that the social world follows a distinct timeline and requires a richer blend of cognitive modes (from intuitive reactions (System 1) and surface-level thinking to deliberate thinking (System 2)) than mathematics, which primarily relies on System 2 cognition (careful, step-by-step reasoning), we introduce Temporal-aware Hierarchical Cognitive Reinforcement Learning (TimeHC-RL) for enhancing LLMs' social intelligence. In our experiments, we systematically explore improving LLMs' social intelligence and validate the effectiveness of the TimeHC-RL method, through five other post-training paradigms and two test-time intervention paradigms on eight datasets with diverse data patterns. Experimental results reveal the superiority of our proposed TimeHC-RL method compared to the widely adopted System 2 RL method. It gives the 7B backbone model wings, enabling it to rival the performance of advanced models like DeepSeek-R1 and OpenAI-O3. Additionally, the systematic exploration from post-training and test-time interventions perspectives to improve LLMs' social intelligence has uncovered several valuable insights. 

---
# Towards Effective Code-Integrated Reasoning 

**Authors**: Fei Bai, Yingqian Min, Beichen Zhang, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Zheng Liu, Zhongyuan Wang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24480)  

**Abstract**: In this paper, we investigate code-integrated reasoning, where models generate code when necessary and integrate feedback by executing it through a code interpreter. To acquire this capability, models must learn when and how to use external code tools effectively, which is supported by tool-augmented reinforcement learning (RL) through interactive learning. Despite its benefits, tool-augmented RL can still suffer from potential instability in the learning dynamics. In light of this challenge, we present a systematic approach to improving the training effectiveness and stability of tool-augmented RL for code-integrated reasoning. Specifically, we develop enhanced training strategies that balance exploration and stability, progressively building tool-use capabilities while improving reasoning performance. Through extensive experiments on five mainstream mathematical reasoning benchmarks, our model demonstrates significant performance improvements over multiple competitive baselines. Furthermore, we conduct an in-depth analysis of the mechanism and effect of code-integrated reasoning, revealing several key insights, such as the extension of model's capability boundaries and the simultaneous improvement of reasoning efficiency through code integration. All data and code for reproducing this work are available at: this https URL. 

---
# VietMix: A Naturally Occurring Vietnamese-English Code-Mixed Corpus with Iterative Augmentation for Machine Translation 

**Authors**: Hieu Tran, Phuong-Anh Nguyen-Le, Huy Nghiem, Quang-Nhan Nguyen, Wei Ai, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2505.24472)  

**Abstract**: Machine translation systems fail when processing code-mixed inputs for low-resource languages. We address this challenge by curating VietMix, a parallel corpus of naturally occurring code-mixed Vietnamese text paired with expert English translations. Augmenting this resource, we developed a complementary synthetic data generation pipeline. This pipeline incorporates filtering mechanisms to ensure syntactic plausibility and pragmatic appropriateness in code-mixing patterns. Experimental validation shows our naturalistic and complementary synthetic data boost models' performance, measured by translation quality estimation scores, of up to 71.84 on COMETkiwi and 81.77 on XCOMET. Triangulating positive results with LLM-based assessments, augmented models are favored over seed fine-tuned counterparts in approximately 49% of judgments (54-56% excluding ties). VietMix and our augmentation methodology advance ecological validity in neural MT evaluations and establish a framework for addressing code-mixed translation challenges across other low-resource pairs. 

---
# CaMMT: Benchmarking Culturally Aware Multimodal Machine Translation 

**Authors**: Emilio Villa-Cueva, Sholpan Bolatzhanova, Diana Turmakhan, Kareem Elzeky, Henok Biadglign Ademtew, Alham Fikri Aji, Israel Abebe Azime, Jinheon Baek, Frederico Belcavello, Fermin Cristobal, Jan Christian Blaise Cruz, Mary Dabre, Raj Dabre, Toqeer Ehsan, Naome A Etori, Fauzan Farooqui, Jiahui Geng, Guido Ivetta, Thanmay Jayakumar, Soyeong Jeong, Zheng Wei Lim, Aishik Mandal, Sofia Martinelli, Mihail Minkov Mihaylov, Daniil Orel, Aniket Pramanick, Sukannya Purkayastha, Israfel Salazar, Haiyue Song, Tiago Timponi Torrent, Debela Desalegn Yadeta, Injy Hamed, Atnafu Lambebo Tonja, Thamar Solorio  

**Link**: [PDF](https://arxiv.org/pdf/2505.24456)  

**Abstract**: Cultural content poses challenges for machine translation systems due to the differences in conceptualizations between cultures, where language alone may fail to convey sufficient context to capture region-specific meanings. In this work, we investigate whether images can act as cultural context in multimodal translation. We introduce CaMMT, a human-curated benchmark of over 5,800 triples of images along with parallel captions in English and regional languages. Using this dataset, we evaluate five Vision Language Models (VLMs) in text-only and text+image settings. Through automatic and human evaluations, we find that visual context generally improves translation quality, especially in handling Culturally-Specific Items (CSIs), disambiguation, and correct gender usage. By releasing CaMMT, we aim to support broader efforts in building and evaluating multimodal translation systems that are better aligned with cultural nuance and regional variation. 

---
# Domain Pre-training Impact on Representations 

**Authors**: Cesar Gonzalez-Gutierrez, Ariadna Quattoni  

**Link**: [PDF](https://arxiv.org/pdf/2505.24455)  

**Abstract**: This empirical study analyzes the effects of the pre-training corpus on the quality of learned transformer representations. We focus on the representation quality induced solely through pre-training. Our experiments show that pre-training on a small, specialized corpus can yield effective representations, and that the success of combining a generic and a specialized corpus depends on the distributional similarity between the target task and the specialized corpus. 

---
# When Large Multimodal Models Confront Evolving Knowledge:Challenges and Pathways 

**Authors**: Kailin Jiang, Yuntao Du, Yukai Ding, Yuchen Ren, Ning Jiang, Zhi Gao, Zilong Zheng, Lei Liu, Bin Li, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24449)  

**Abstract**: Large language/multimodal models (LLMs/LMMs) store extensive pre-trained knowledge but struggle to maintain consistency with real-world updates, making it difficult to avoid catastrophic forgetting while acquiring evolving knowledge. Previous work focused on constructing textual knowledge datasets and exploring knowledge injection in LLMs, lacking exploration of multimodal evolving knowledge injection in LMMs. To address this, we propose the EVOKE benchmark to evaluate LMMs' ability to inject multimodal evolving knowledge in real-world scenarios. Meanwhile, a comprehensive evaluation of multimodal evolving knowledge injection revealed two challenges: (1) Existing knowledge injection methods perform terribly on evolving knowledge. (2) Supervised fine-tuning causes catastrophic forgetting, particularly instruction following ability is severely compromised. Additionally, we provide pathways and find that: (1) Text knowledge augmentation during the training phase improves performance, while image augmentation cannot achieve it. (2) Continual learning methods, especially Replay and MoELoRA, effectively mitigate forgetting. Our findings indicate that current knowledge injection methods have many limitations on evolving knowledge, which motivates further research on more efficient and stable knowledge injection methods. 

---
# Exploring the Impact of Occupational Personas on Domain-Specific QA 

**Authors**: Eojin Kang, Jaehyuk Yu, Juae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.24448)  

**Abstract**: Recent studies on personas have improved the way Large Language Models (LLMs) interact with users. However, the effect of personas on domain-specific question-answering (QA) tasks remains a subject of debate. This study analyzes whether personas enhance specialized QA performance by introducing two types of persona: Profession-Based Personas (PBPs) (e.g., scientist), which directly relate to domain expertise, and Occupational Personality-Based Personas (OPBPs) (e.g., scientific person), which reflect cognitive tendencies rather than explicit expertise. Through empirical evaluations across multiple scientific domains, we demonstrate that while PBPs can slightly improve accuracy, OPBPs often degrade performance, even when semantically related to the task. Our findings suggest that persona relevance alone does not guarantee effective knowledge utilization and that they may impose cognitive constraints that hinder optimal knowledge application. Future research can explore how nuanced distinctions in persona representations guide LLMs, potentially contributing to reasoning and knowledge retrieval that more closely mirror human social conceptualization. 

---
# Model Unlearning via Sparse Autoencoder Subspace Guided Projections 

**Authors**: Xu Wang, Zihao Li, Benyou Wang, Yan Hu, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.24428)  

**Abstract**: Large language models (LLMs) store vast amounts of information, making them powerful yet raising privacy and safety concerns when selective knowledge removal is required. Existing unlearning strategies, ranging from gradient-based fine-tuning and model editing to sparse autoencoder (SAE) steering, either lack interpretability or fail to provide a robust defense against adversarial prompts. We propose SAE-Guided Subspace Projection Unlearning (SSPU), a novel framework that leverages SAE features to drive targeted updates in the model's parameter space, enabling precise, interpretable, and robust unlearning. SSPU's three-stage pipeline performs data-driven layer and feature selection, subspace construction via QR decomposition, and constrained optimization that controls activations into an "irrelevant" subspace while preserving retained knowledge. Overall, we use SAE features to construct a subspace that supervises unlearning, refining the loss and adding a regularization term to guide interpretable parameter updates. In experiments on the WMDP-Cyber forget set and three utility benchmarks (MMLU, TruthfulQA, GSM8K), SSPU reduces harmful knowledge accuracy by 3.22% compared to the strongest baseline. It also improves adversarial robustness, lowering malicious accuracy under jailbreak prompts compared to baselines. Our findings expose the limitations of prior unlearning methods and demonstrate how interpretable subspace-guided optimization can achieve robust, controllable model behavior. 

---
# Donate or Create? Comparing Data Collection Strategies for Emotion-labeled Multimodal Social Media Posts 

**Authors**: Christopher Bagdon, Aidan Combs, Carina Silberer, Roman Klinger  

**Link**: [PDF](https://arxiv.org/pdf/2505.24427)  

**Abstract**: Accurate modeling of subjective phenomena such as emotion expression requires data annotated with authors' intentions. Commonly such data is collected by asking study participants to donate and label genuine content produced in the real world, or create content fitting particular labels during the study. Asking participants to create content is often simpler to implement and presents fewer risks to participant privacy than data donation. However, it is unclear if and how study-created content may differ from genuine content, and how differences may impact models. We collect study-created and genuine multimodal social media posts labeled for emotion and compare them on several dimensions, including model performance. We find that compared to genuine posts, study-created posts are longer, rely more on their text and less on their images for emotion expression, and focus more on emotion-prototypical events. The samples of participants willing to donate versus create posts are demographically different. Study-created data is valuable to train models that generalize well to genuine data, but realistic effectiveness estimates require genuine data. 

---
# MMAFFBen: A Multilingual and Multimodal Affective Analysis Benchmark for Evaluating LLMs and VLMs 

**Authors**: Zhiwei Liu, Lingfei Qian, Qianqian Xie, Jimin Huang, Kailai Yang, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2505.24423)  

**Abstract**: Large language models and vision-language models (which we jointly call LMs) have transformed NLP and CV, demonstrating remarkable potential across various fields. However, their capabilities in affective analysis (i.e. sentiment analysis and emotion detection) remain underexplored. This gap is largely due to the absence of comprehensive evaluation benchmarks, and the inherent complexity of affective analysis tasks. In this paper, we introduce MMAFFBen, the first extensive open-source benchmark for multilingual multimodal affective analysis. MMAFFBen encompasses text, image, and video modalities across 35 languages, covering four key affective analysis tasks: sentiment polarity, sentiment intensity, emotion classification, and emotion intensity. Moreover, we construct the MMAFFIn dataset for fine-tuning LMs on affective analysis tasks, and further develop MMAFFLM-3b and MMAFFLM-7b based on it. We evaluate various representative LMs, including GPT-4o-mini, providing a systematic comparison of their affective understanding capabilities. This project is available at this https URL. 

---
# LLMs Are Globally Multilingual Yet Locally Monolingual: Exploring Knowledge Transfer via Language and Thought Theory 

**Authors**: Eojin Kang, Juae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.24409)  

**Abstract**: Multilingual large language models (LLMs) open up new possibilities for leveraging information across languages, but their factual knowledge recall remains inconsistent depending on the input language. While previous studies have attempted to address this issue through English-based prompting and evaluation, we explore non-English to English transfer via Language and Thought Theory. This perspective allows us to examine language-thought binding in LLMs and uncover why factual knowledge often fails to transfer effectively. We propose the Language-to-Thought (L2T) prompting strategy, which analyzes the relationship between input language, internal cognitive processes, and knowledge. Experimental results challenge the assumption that English-based approaches consistently outperform other languages and offer a novel insight that aligning the model's internal thought with the knowledge required for the task is critical for successful cross-lingual transfer. Furthermore, we show that applying L2T during training can alleviate LLMs' reliance on the input language and facilitate cross-linguistic knowledge integration without translation-based learning. Code and datasets will be available. 

---
# ClueAnchor: Clue-Anchored Knowledge Reasoning Exploration and Optimization for Retrieval-Augmented Generation 

**Authors**: Hao Chen, Yukun Yan, Sen Mei, Wanxiang Che, Zhenghao Liu, Qi Shi, Xinze Li, Yuchun Fan, Pengcheng Huang, Qiushi Xiong, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.24388)  

**Abstract**: Retrieval-Augmented Generation (RAG) augments Large Language Models (LLMs) with external knowledge to improve factuality. However, existing RAG systems frequently underutilize the retrieved documents, failing to extract and integrate the key clues needed to support faithful and interpretable reasoning, especially in cases where relevant evidence is implicit, scattered, or obscured by noise. To address this issue, we propose ClueAnchor, a novel framework for enhancing RAG via clue-anchored reasoning exploration and optimization. ClueAnchor extracts key clues from retrieved content and generates multiple reasoning paths based on different knowledge configurations, optimizing the model by selecting the most effective one through reward-based preference optimization. Experiments show that ClueAnchor significantly outperforms prior RAG baselines in reasoning completeness and robustness. Further analysis confirms its strong resilience to noisy or partially relevant retrieved content, as well as its capability to identify supporting evidence even in the absence of explicit clue supervision during inference. 

---
# LLM Inference Enhanced by External Knowledge: A Survey 

**Authors**: Yu-Hsuan Lin, Qian-Hui Chen, Yi-Jie Cheng, Jia-Ren Zhang, Yi-Hung Liu, Liang-Yu Hsia, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24377)  

**Abstract**: Recent advancements in large language models (LLMs) have enhanced natural-language reasoning. However, their limited parametric memory and susceptibility to hallucination present persistent challenges for tasks requiring accurate, context-based inference. To overcome these limitations, an increasing number of studies have proposed leveraging external knowledge to enhance LLMs. This study offers a systematic exploration of strategies for using external knowledge to enhance LLMs, beginning with a taxonomy that categorizes external knowledge into unstructured and structured data. We then focus on structured knowledge, presenting distinct taxonomies for tables and knowledge graphs (KGs), detailing their integration paradigms with LLMs, and reviewing representative methods. Our comparative analysis further highlights the trade-offs among interpretability, scalability, and performance, providing insights for developing trustworthy and generalizable knowledge-enhanced LLMs. 

---
# Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion 

**Authors**: Anum Afzal, Florian Matthes, Gal Chechik, Yftah Ziser  

**Link**: [PDF](https://arxiv.org/pdf/2505.24362)  

**Abstract**: We investigate whether the success of a zero-shot Chain-of-Thought (CoT) process can be predicted before completion. We discover that a probing classifier, based on LLM representations, performs well \emph{even before a single token is generated}, suggesting that crucial information about the reasoning process is already present in the initial steps representations. In contrast, a strong BERT-based baseline, which relies solely on the generated tokens, performs worse, likely because it depends on shallow linguistic cues rather than deeper reasoning dynamics. Surprisingly, using later reasoning steps does not always improve classification. When additional context is unhelpful, earlier representations resemble later ones more, suggesting LLMs encode key information early. This implies reasoning can often stop early without loss. To test this, we conduct early stopping experiments, showing that truncating CoT reasoning still improves performance over not using CoT at all, though a gap remains compared to full reasoning. However, approaches like supervised learning or reinforcement learning designed to shorten CoT chains could leverage our classifier's guidance to identify when early stopping is effective. Our findings provide insights that may support such methods, helping to optimize CoT's efficiency while preserving its benefits.\footnote{Code and data is available at \href{this https URL}{\texttt{this http URL}}. 

---
# Multilingual Gloss-free Sign Language Translation: Towards Building a Sign Language Foundation Model 

**Authors**: Sihan Tan, Taro Miyazaki, Kazuhiro Nakadai  

**Link**: [PDF](https://arxiv.org/pdf/2505.24355)  

**Abstract**: Sign Language Translation (SLT) aims to convert sign language (SL) videos into spoken language text, thereby bridging the communication gap between the sign and the spoken community. While most existing works focus on translating a single sign language into a single spoken language (one-to-one SLT), leveraging multilingual resources could mitigate low-resource issues and enhance accessibility. However, multilingual SLT (MLSLT) remains unexplored due to language conflicts and alignment difficulties across SLs and spoken languages. To address these challenges, we propose a multilingual gloss-free model with dual CTC objectives for token-level SL identification and spoken text generation. Our model supports 10 SLs and handles one-to-one, many-to-one, and many-to-many SLT tasks, achieving competitive performance compared to state-of-the-art methods on three widely adopted benchmarks: multilingual SP-10, PHOENIX14T, and CSL-Daily. 

---
# Unifying Language Agent Algorithms with Graph-based Orchestration Engine for Reproducible Agent Research 

**Authors**: Qianqian Zhang, Jiajia Liao, Heting Ying, Yibo Ma, Haozhan Shen, Jingcheng Li, Peng Liu, Lu Zhang, Chunxin Fang, Kyusong Lee, Ruochen Xu, Tiancheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.24354)  

**Abstract**: Language agents powered by large language models (LLMs) have demonstrated remarkable capabilities in understanding, reasoning, and executing complex tasks. However, developing robust agents presents significant challenges: substantial engineering overhead, lack of standardized components, and insufficient evaluation frameworks for fair comparison. We introduce Agent Graph-based Orchestration for Reasoning and Assessment (AGORA), a flexible and extensible framework that addresses these challenges through three key contributions: (1) a modular architecture with a graph-based workflow engine, efficient memory management, and clean component abstraction; (2) a comprehensive suite of reusable agent algorithms implementing state-of-the-art reasoning approaches; and (3) a rigorous evaluation framework enabling systematic comparison across multiple dimensions. Through extensive experiments on mathematical reasoning and multimodal tasks, we evaluate various agent algorithms across different LLMs, revealing important insights about their relative strengths and applicability. Our results demonstrate that while sophisticated reasoning approaches can enhance agent capabilities, simpler methods like Chain-of-Thought often exhibit robust performance with significantly lower computational overhead. AGORA not only simplifies language agent development but also establishes a foundation for reproducible agent research through standardized evaluation protocols. 

---
# Fewer Hallucinations, More Verification: A Three-Stage LLM-Based Framework for ASR Error Correction 

**Authors**: Yangui Fang, Baixu Cheng, Jing Peng, Xu Li, Yu Xi, Chengwei Zhang, Guohui Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2505.24347)  

**Abstract**: Automatic Speech Recognition (ASR) error correction aims to correct recognition errors while preserving accurate text. Although traditional approaches demonstrate moderate effectiveness, LLMs offer a paradigm that eliminates the need for training and labeled data. However, directly using LLMs will encounter hallucinations problem, which may lead to the modification of the correct text. To address this problem, we propose the Reliable LLM Correction Framework (RLLM-CF), which consists of three stages: (1) error pre-detection, (2) chain-of-thought sub-tasks iterative correction, and (3) reasoning process verification. The advantage of our method is that it does not require additional information or fine-tuning of the model, and ensures the correctness of the LLM correction under multi-pass programming. Experiments on AISHELL-1, AISHELL-2, and Librispeech show that the GPT-4o model enhanced by our framework achieves 21%, 11%, 9%, and 11.4% relative reductions in CER/WER. 

---
# Exploring Multimodal Challenges in Toxic Chinese Detection: Taxonomy, Benchmark, and Findings 

**Authors**: Shujian Yang, Shiyao Cui, Chuanrui Hu, Haicheng Wang, Tianwei Zhang, Minlie Huang, Jialiang Lu, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24341)  

**Abstract**: Detecting toxic content using language models is important but challenging. While large language models (LLMs) have demonstrated strong performance in understanding Chinese, recent studies show that simple character substitutions in toxic Chinese text can easily confuse the state-of-the-art (SOTA) LLMs. In this paper, we highlight the multimodal nature of Chinese language as a key challenge for deploying LLMs in toxic Chinese detection. First, we propose a taxonomy of 3 perturbation strategies and 8 specific approaches in toxic Chinese content. Then, we curate a dataset based on this taxonomy, and benchmark 9 SOTA LLMs (from both the US and China) to assess if they can detect perturbed toxic Chinese text. Additionally, we explore cost-effective enhancement solutions like in-context learning (ICL) and supervised fine-tuning (SFT). Our results reveal two important findings. (1) LLMs are less capable of detecting perturbed multimodal Chinese toxic contents. (2) ICL or SFT with a small number of perturbed examples may cause the LLMs "overcorrect'': misidentify many normal Chinese contents as toxic. 

---
# Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning 

**Authors**: Wenxuan Shi, Haochen Tan, Chuqiao Kuang, Xiaoguang Li, Xiaozhe Ren, Chen Zhang, Hanting Chen, Yasheng Wang, Lifeng Shang, Fisher Yu, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24332)  

**Abstract**: Information seeking demands iterative evidence gathering and reflective reasoning, yet large language models (LLMs) still struggle with it in open-web question answering. Existing methods rely on static prompting rules or training with Wikipedia-based corpora and retrieval environments, limiting adaptability to the real-world web environment where ambiguity, conflicting evidence, and noise are prevalent. These constrained training settings hinder LLMs from learning to dynamically decide when and where to search, and how to adjust search depth and frequency based on informational demands. We define this missing capacity as Search Intensity Scaling (SIS)--the emergent skill to intensify search efforts under ambiguous or conflicting conditions, rather than settling on overconfident, under-verification answers.
To study SIS, we introduce WebPuzzle, the first dataset designed to foster information-seeking behavior in open-world internet environments. WebPuzzle consists of 24K training instances and 275 test questions spanning both wiki-based and open-web queries. Building on this dataset, we propose DeepDiver, a Reinforcement Learning (RL) framework that promotes SIS by encouraging adaptive search policies through exploration under a real-world open-web environment. Experimental results show that Pangu-7B-Reasoner empowered by DeepDiver achieve performance on real-web tasks comparable to the 671B-parameter DeepSeek-R1. We detail DeepDiver's training curriculum from cold-start supervised fine-tuning to a carefully designed RL phase, and present that its capability of SIS generalizes from closed-form QA to open-ended tasks such as long-form writing. Our contributions advance adaptive information seeking in LLMs and provide a valuable benchmark and dataset for future research. 

---
# Context-Aware Sentiment Forecasting via LLM-based Multi-Perspective Role-Playing Agents 

**Authors**: Fanhang Man, Huandong Wang, Jianjie Fang, Zhaoyi Deng, Baining Zhao, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24331)  

**Abstract**: User sentiment on social media reveals the underlying social trends, crises, and needs. Researchers have analyzed users' past messages to trace the evolution of sentiments and reconstruct sentiment dynamics. However, predicting the imminent sentiment of an ongoing event is rarely studied. In this paper, we address the problem of \textbf{sentiment forecasting} on social media to predict the user's future sentiment in response to the development of the event. We extract sentiment-related features to enhance the modeling skill and propose a multi-perspective role-playing framework to simulate the process of human response. Our preliminary results show significant improvement in sentiment forecasting on both microscopic and macroscopic levels. 

---
# HiCaM: A Hierarchical-Causal Modification Framework for Long-Form Text Modification 

**Authors**: Yuntao Shi, Yi Luo, Yeyun Gong, Chen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.24319)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in various domains. However, when handling long-form text modification tasks, they still face two major problems: (1) producing undesired modifications by inappropriately altering or summarizing irrelevant content, and (2) missing necessary modifications to implicitly related passages that are crucial for maintaining document coherence. To address these issues, we propose HiCaM, a Hierarchical-Causal Modification framework that operates through a hierarchical summary tree and a causal graph. Furthermore, to evaluate HiCaM, we derive a multi-domain dataset from various benchmarks, providing a resource for assessing its effectiveness. Comprehensive evaluations on the dataset demonstrate significant improvements over strong LLMs, with our method achieving up to a 79.50\% win rate. These results highlight the comprehensiveness of our approach, showing consistent performance improvements across multiple models and domains. 

---
# ScienceMeter: Tracking Scientific Knowledge Updates in Language Models 

**Authors**: Yike Wang, Shangbin Feng, Yulia Tsvetkov, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2505.24302)  

**Abstract**: Large Language Models (LLMs) are increasingly used to support scientific research, but their knowledge of scientific advancements can quickly become outdated. We introduce ScienceMeter, a new framework for evaluating scientific knowledge update methods over scientific knowledge spanning the past, present, and future. ScienceMeter defines three metrics: knowledge preservation, the extent to which models' understanding of previously learned papers are preserved; knowledge acquisition, how well scientific claims from newly introduced papers are acquired; and knowledge projection, the ability of the updated model to anticipate or generalize to related scientific claims that may emerge in the future. Using ScienceMeter, we examine the scientific knowledge of LLMs on claim judgment and generation tasks across a curated dataset of 15,444 scientific papers and 30,888 scientific claims from ten domains including medicine, biology, materials science, and computer science. We evaluate five representative knowledge update approaches including training- and inference-time methods. With extensive experiments, we find that the best-performing knowledge update methods can preserve only 85.9% of existing knowledge, acquire 71.7% of new knowledge, and project 37.7% of future knowledge. Inference-based methods work for larger models, whereas smaller models require training to achieve comparable performance. Cross-domain analysis reveals that performance on these objectives is correlated. Even when applying on specialized scientific LLMs, existing knowledge update methods fail to achieve these objectives collectively, underscoring that developing robust scientific knowledge update mechanisms is both crucial and challenging. 

---
# Faithful and Robust LLM-Driven Theorem Proving for NLI Explanations 

**Authors**: Xin Quan, Marco Valentino, Louise A. Dennis, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.24264)  

**Abstract**: Natural language explanations play a fundamental role in Natural Language Inference (NLI) by revealing how premises logically entail hypotheses. Recent work has shown that the interaction of large language models (LLMs) with theorem provers (TPs) can help verify and improve the validity of NLI explanations. However, TPs require translating natural language into machine-verifiable formal representations, a process that introduces the risk of semantic information loss and unfaithful interpretation, an issue compounded by LLMs' challenges in capturing critical logical structures with sufficient precision. Moreover, LLMs are still limited in their capacity for rigorous and robust proof construction within formal verification frameworks. To mitigate issues related to faithfulness and robustness, this paper investigates strategies to (1) alleviate semantic loss during autoformalisation, (2) efficiently identify and correct syntactic errors in logical representations, (3) explicitly use logical expressions to guide LLMs in generating structured proof sketches, and (4) increase LLMs' capacity of interpreting TP's feedback for iterative refinement. Our empirical results on e-SNLI, QASC and WorldTree using different LLMs demonstrate that the proposed strategies yield significant improvements in autoformalisation (+18.46%, +34.2%, +39.77%) and explanation refinement (+29.5%, +51.5%, +41.25%) over the state-of-the-art model. Moreover, we show that specific interventions on the hybrid LLM-TP architecture can substantially improve efficiency, drastically reducing the number of iterations required for successful verification. 

---
# Simulating Training Data Leakage in Multiple-Choice Benchmarks for LLM Evaluation 

**Authors**: Naila Shafirni Hidayat, Muhammad Dehan Al Kautsar, Alfan Farizki Wicaksono, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2505.24263)  

**Abstract**: The performance of large language models (LLMs) continues to improve, as reflected in rising scores on standard benchmarks. However, the lack of transparency around training data raises concerns about potential overlap with evaluation sets and the fairness of reported results. Although prior work has proposed methods for detecting data leakage, these approaches primarily focus on identifying outliers and have not been evaluated under controlled simulated leakage conditions. In this work, we compare existing leakage detection techniques, namely permutation and n-gram-based methods, under a continual pretraining setup that simulates real-world leakage scenarios, and additionally explore a lightweight method we call semi-half question. Although semi-half offers a low-cost alternative, our analysis shows that the n-gram method consistently achieves the highest F1-score. We also refine these techniques to support instance-level detection and reduce computational overhead. Leveraging the best-performing method, we create cleaned versions of MMLU and HellaSwag, and re-evaluate several LLMs. Our findings present a practical path toward more reliable and transparent evaluations, and we recommend contamination checks as a standard step before releasing benchmark results. 

---
# Effects of Theory of Mind and Prosocial Beliefs on Steering Human-Aligned Behaviors of LLMs in Ultimatum Games 

**Authors**: Neemesh Yadav, Palakorn Achananuparp, Jing Jiang, Ee-Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2505.24255)  

**Abstract**: Large Language Models (LLMs) have shown potential in simulating human behaviors and performing theory-of-mind (ToM) reasoning, a crucial skill for complex social interactions. In this study, we investigate the role of ToM reasoning in aligning agentic behaviors with human norms in negotiation tasks, using the ultimatum game as a controlled environment. We initialized LLM agents with different prosocial beliefs (including Greedy, Fair, and Selfless) and reasoning methods like chain-of-thought (CoT) and varying ToM levels, and examined their decision-making processes across diverse LLMs, including reasoning models like o3-mini and DeepSeek-R1 Distilled Qwen 32B. Results from 2,700 simulations indicated that ToM reasoning enhances behavior alignment, decision-making consistency, and negotiation outcomes. Consistent with previous findings, reasoning models exhibit limited capability compared to models with ToM reasoning, different roles of the game benefits with different orders of ToM reasoning. Our findings contribute to the understanding of ToM's role in enhancing human-AI interaction and cooperative decision-making. The code used for our experiments can be found at this https URL. 

---
# Proactive Guidance of Multi-Turn Conversation in Industrial Search 

**Authors**: Xiaoyu Li, Xiao Li, Li Gao, Yiding Liu, Xiaoyang Wang, Shuaiqiang Wang, Junfeng Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.24251)  

**Abstract**: The evolution of Large Language Models (LLMs) has significantly advanced multi-turn conversation systems, emphasizing the need for proactive guidance to enhance users' interactions. However, these systems face challenges in dynamically adapting to shifts in users' goals and maintaining low latency for real-time interactions. In the Baidu Search AI assistant, an industrial-scale multi-turn search system, we propose a novel two-phase framework to provide proactive guidance. The first phase, Goal-adaptive Supervised Fine-Tuning (G-SFT), employs a goal adaptation agent that dynamically adapts to user goal shifts and provides goal-relevant contextual information. G-SFT also incorporates scalable knowledge transfer to distill insights from LLMs into a lightweight model for real-time interaction. The second phase, Click-oriented Reinforcement Learning (C-RL), adopts a generate-rank paradigm, systematically constructs preference pairs from user click signals, and proactively improves click-through rates through more engaging guidance. This dual-phase architecture achieves complementary objectives: G-SFT ensures accurate goal tracking, while C-RL optimizes interaction quality through click signal-driven reinforcement learning. Extensive experiments demonstrate that our framework achieves 86.10% accuracy in offline evaluation (+23.95% over baseline) and 25.28% CTR in online deployment (149.06% relative improvement), while reducing inference latency by 69.55% through scalable knowledge distillation. 

---
# Mamba Knockout for Unraveling Factual Information Flow 

**Authors**: Nir Endy, Idan Daniel Grosbard, Yuval Ran-Milo, Yonatan Slutzky, Itay Tshuva, Raja Giryes  

**Link**: [PDF](https://arxiv.org/pdf/2505.24244)  

**Abstract**: This paper investigates the flow of factual information in Mamba State-Space Model (SSM)-based language models. We rely on theoretical and empirical connections to Transformer-based architectures and their attention mechanisms. Exploiting this relationship, we adapt attentional interpretability techniques originally developed for Transformers--specifically, the Attention Knockout methodology--to both Mamba-1 and Mamba-2. Using them we trace how information is transmitted and localized across tokens and layers, revealing patterns of subject-token information emergence and layer-wise dynamics. Notably, some phenomena vary between mamba models and Transformer based models, while others appear universally across all models inspected--hinting that these may be inherent to LLMs in general. By further leveraging Mamba's structured factorization, we disentangle how distinct "features" either enable token-to-token information exchange or enrich individual tokens, thus offering a unified lens to understand Mamba internal operations. 

---
# Advantageous Parameter Expansion Training Makes Better Large Language Models 

**Authors**: Naibin Gu, Yilong Chen, Zhenyu Zhang, Peng Fu, Zheng Lin, Shuohuan Wang, Yu Sun, Hua Wu, Weiping Wang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24241)  

**Abstract**: Although scaling up the number of trainable parameters in both pre-training and fine-tuning can effectively improve the performance of large language models, it also leads to increased computational overhead. When delving into the parameter difference, we find that a subset of parameters, termed advantageous parameters, plays a crucial role in determining model performance. Further analysis reveals that stronger models tend to possess more such parameters. In this paper, we propose Advantageous Parameter EXpansion Training (APEX), a method that progressively expands advantageous parameters into the space of disadvantageous ones, thereby increasing their proportion and enhancing training effectiveness. Further theoretical analysis from the perspective of matrix effective rank explains the performance gains of APEX. Extensive experiments on both instruction tuning and continued pre-training demonstrate that, in instruction tuning, APEX outperforms full-parameter tuning while using only 52% of the trainable parameters. In continued pre-training, APEX achieves the same perplexity level as conventional training with just 33% of the training data, and yields significant improvements on downstream tasks. 

---
# Dynamic Context-Aware Streaming Pretrained Language Model For Inverse Text Normalization 

**Authors**: Luong Ho, Khanh Le, Vinh Pham, Bao Nguyen, Tan Tran, Duc Chau  

**Link**: [PDF](https://arxiv.org/pdf/2505.24229)  

**Abstract**: Inverse Text Normalization (ITN) is crucial for converting spoken Automatic Speech Recognition (ASR) outputs into well-formatted written text, enhancing both readability and usability. Despite its importance, the integration of streaming ITN within streaming ASR remains largely unexplored due to challenges in accuracy, efficiency, and adaptability, particularly in low-resource and limited-context scenarios. In this paper, we introduce a streaming pretrained language model for ITN, leveraging pretrained linguistic representations for improved robustness. To address streaming constraints, we propose Dynamic Context-Aware during training and inference, enabling adaptive chunk size adjustments and the integration of right-context information. Experimental results demonstrate that our method achieves accuracy comparable to non-streaming ITN and surpasses existing streaming ITN models on a Vietnamese dataset, all while maintaining low latency, ensuring seamless integration into ASR systems. 

---
# Automated Structured Radiology Report Generation 

**Authors**: Jean-Benoit Delbrouck, Justin Xu, Johannes Moll, Alois Thomas, Zhihong Chen, Sophie Ostmeier, Asfandyar Azhar, Kelvin Zhenghao Li, Andrew Johnston, Christian Bluethgen, Eduardo Reis, Mohamed Muneer, Maya Varma, Curtis Langlotz  

**Link**: [PDF](https://arxiv.org/pdf/2505.24223)  

**Abstract**: Automated radiology report generation from chest X-ray (CXR) images has the potential to improve clinical efficiency and reduce radiologists' workload. However, most datasets, including the publicly available MIMIC-CXR and CheXpert Plus, consist entirely of free-form reports, which are inherently variable and unstructured. This variability poses challenges for both generation and evaluation: existing models struggle to produce consistent, clinically meaningful reports, and standard evaluation metrics fail to capture the nuances of radiological interpretation. To address this, we introduce Structured Radiology Report Generation (SRRG), a new task that reformulates free-text radiology reports into a standardized format, ensuring clarity, consistency, and structured clinical reporting. We create a novel dataset by restructuring reports using large language models (LLMs) following strict structured reporting desiderata. Additionally, we introduce SRR-BERT, a fine-grained disease classification model trained on 55 labels, enabling more precise and clinically informed evaluation of structured reports. To assess report quality, we propose F1-SRR-BERT, a metric that leverages SRR-BERT's hierarchical disease taxonomy to bridge the gap between free-text variability and structured clinical reporting. We validate our dataset through a reader study conducted by five board-certified radiologists and extensive benchmarking experiments. 

---
# ERU-KG: Efficient Reference-aligned Unsupervised Keyphrase Generation 

**Authors**: Lam Thanh Do, Aaditya Bodke, Pritom Saha Akash, Kevin Chen-Chuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24219)  

**Abstract**: Unsupervised keyphrase prediction has gained growing interest in recent years. However, existing methods typically rely on heuristically defined importance scores, which may lead to inaccurate informativeness estimation. In addition, they lack consideration for time efficiency. To solve these problems, we propose ERU-KG, an unsupervised keyphrase generation (UKG) model that consists of an informativeness and a phraseness module. The former estimates the relevance of keyphrase candidates, while the latter generate those candidates. The informativeness module innovates by learning to model informativeness through references (e.g., queries, citation contexts, and titles) and at the term-level, thereby 1) capturing how the key concepts of documents are perceived in different contexts and 2) estimating informativeness of phrases more efficiently by aggregating term informativeness, removing the need for explicit modeling of the candidates. ERU-KG demonstrates its effectiveness on keyphrase generation benchmarks by outperforming unsupervised baselines and achieving on average 89\% of the performance of a supervised model for top 10 predictions. Additionally, to highlight its practical utility, we evaluate the model on text retrieval tasks and show that keyphrases generated by ERU-KG are effective when employed as query and document expansions. Furthermore, inference speed tests reveal that ERU-KG is the fastest among baselines of similar model sizes. Finally, our proposed model can switch between keyphrase generation and extraction by adjusting hyperparameters, catering to diverse application requirements. 

---
# Semi-structured LLM Reasoners Can Be Rigorously Audited 

**Authors**: Jixuan Leng, Cassandra A. Cohen, Zhixian Zhang, Chenyan Xiong, William W. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24217)  

**Abstract**: As Large Language Models (LLMs) become increasingly capable at reasoning, the problem of "faithfulness" persists: LLM "reasoning traces" can contain errors and omissions that are difficult to detect, and may obscure biases in model outputs. To address these limitations, we introduce Semi-Structured Reasoning Models (SSRMs), which internalize a semi-structured Chain-of-Thought (CoT) reasoning format within the model. Our SSRMs generate reasoning traces in a Pythonic syntax. While SSRM traces are not executable, they adopt a restricted, task-specific vocabulary to name distinct reasoning steps, and to mark each step's inputs and outputs. Through extensive evaluation on ten benchmarks, SSRMs demonstrate strong performance and generality: they outperform comparably sized baselines by nearly ten percentage points on in-domain tasks while remaining competitive with specialized models on out-of-domain medical benchmarks. Furthermore, we show that semi-structured reasoning is more amenable to analysis: in particular, they can be automatically audited to identify reasoning flaws. We explore both hand-crafted structured audits, which detect task-specific problematic reasoning patterns, and learned typicality audits, which apply probabilistic models over reasoning patterns, and show that both audits can be used to effectively flag probable reasoning errors. 

---
# Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists? 

**Authors**: Jiwan Chung, Janghan Yoon, Junhyeong Park, Sangeyl Lee, Joowon Yang, Sooyeon Park, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24211)  

**Abstract**: Any-to-any generative models aim to enable seamless interpretation and generation across multiple modalities within a unified framework, yet their ability to preserve relationships across modalities remains uncertain. Do unified models truly achieve cross-modal coherence, or is this coherence merely perceived? To explore this, we introduce ACON, a dataset of 1,000 images (500 newly contributed) paired with captions, editing instructions, and Q&A pairs to evaluate cross-modal transfers rigorously. Using three consistency criteria-cyclic consistency, forward equivariance, and conjugated equivariance-our experiments reveal that any-to-any models do not consistently demonstrate greater cross-modal consistency than specialized models in pointwise evaluations such as cyclic consistency. However, equivariance evaluations uncover weak but observable consistency through structured analyses of the intermediate latent space enabled by multiple editing operations. We release our code and data at this https URL. 

---
# Intuitionistic Fuzzy Sets for Large Language Model Data Annotation: A Novel Approach to Side-by-Side Preference Labeling 

**Authors**: Yimin Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.24199)  

**Abstract**: The quality of human preference data is crucial for training and evaluating large language models (LLMs), particularly in reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO) scenarios. Traditional side-by-side (SBS) annotation approaches often struggle with inherent uncertainty, annotator disagreement, and the complexity of preference judgments. This paper introduces a novel framework based on intuitionistic fuzzy sets (IFS) for modeling and aggregating human preferences in LLM data annotation tasks. Our approach captures not only the degree of preference but also the uncertainty and hesitation inherent in human judgment through membership, non-membership, and hesitation degrees. We propose an IFS-based annotation protocol that enables more nuanced preference modeling, develops aggregation methods for handling annotator disagreement, and introduces quality metrics for preference data assessment. Experimental validation on multiple datasets demonstrates that our IFS-based approach significantly improves annotation consistency, reduces annotator fatigue, and produces higher-quality preference data compared to traditional binary and Likert-scale methods. The resulting preference datasets lead to improved model performance in downstream tasks, with 12.3\% improvement in win-rate against baseline models and 15.7\% reduction in annotation time. Our framework provides a principled approach to handling uncertainty in human preference annotation and offers practical benefits for large-scale LLM training. 

---
# CLaSp: In-Context Layer Skip for Self-Speculative Decoding 

**Authors**: Longze Chen, Renke Shan, Huiming Wang, Lu Wang, Ziqiang Liu, Run Luo, Jiawei Wang, Hamid Alinejad-Rokny, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24196)  

**Abstract**: Speculative decoding (SD) is a promising method for accelerating the decoding process of Large Language Models (LLMs). The efficiency of SD primarily hinges on the consistency between the draft model and the verify model. However, existing drafting approaches typically require additional modules to be trained, which can be challenging to implement and ensure compatibility across various LLMs. In this paper, we propose CLaSp, an in-context layer-skipping strategy for self-speculative decoding. Unlike prior methods, CLaSp does not require additional drafting modules or extra training. Instead, it employs a plug-and-play mechanism by skipping intermediate layers of the verify model to construct a compressed draft model. Specifically, we develop a dynamic programming algorithm that optimizes the layer-skipping process by leveraging the complete hidden states from the last verification stage as an objective. This enables CLaSp to dynamically adjust its layer-skipping strategy after each verification stage, without relying on pre-optimized sets of skipped layers. Experimental results across diverse downstream tasks demonstrate that CLaSp achieves a speedup of 1.3x ~ 1.7x on LLaMA3 series models without altering the original distribution of the generated text. 

---
# Beyond Exponential Decay: Rethinking Error Accumulation in Large Language Models 

**Authors**: Mikhail L. Arbuzov, Alexey A. Shvets, Sisong Beir  

**Link**: [PDF](https://arxiv.org/pdf/2505.24187)  

**Abstract**: The prevailing assumption of an exponential decay in large language model (LLM) reliability with sequence length, predicated on independent per-token error probabilities, posits an inherent limitation for long autoregressive outputs. Our research fundamentally challenges this view by synthesizing emerging evidence that LLM errors are not uniformly distributed but are concentrated at sparse "key tokens" ($5-10\%$ of total tokens) representing critical decision junctions. By distinguishing these high-impact tokens from the increasingly predictable majority, we introduce a new reliability formula explaining the sustained coherence of modern LLMs over thousands of tokens. Converging research streams reveal that long-context performance primarily depends on accurately navigating a few crucial semantic decision points rather than on uniform token-level accuracy, enabling targeted strategies that significantly outperform brute-force approaches. We thus propose a framework for next-generation systems centered on selective preservation of semantically vital tokens, dynamic computational allocation at uncertain decision boundaries, multi-path exploration at ambiguities, and architectures aligned with natural semantic domains. This marks a fundamental shift from raw scaling to strategic reasoning, promising breakthrough performance without proportionate computational scaling and offering a more nuanced understanding that supersedes the exponential decay hypothesis, thereby opening pathways toward substantially more powerful and efficient language systems. 

---
# Adaptive LoRA Merge with Parameter Pruning for Low-Resource Generation 

**Authors**: Ryota Miyano, Yuki Arase  

**Link**: [PDF](https://arxiv.org/pdf/2505.24174)  

**Abstract**: This study proposes a simple yet effective LoRA merge method to achieve LLM adaptation for low-resource language generation tasks. The LoRA merge technique, which integrates multiple LoRA modules trained on different tasks, has gained attention as an effective and efficient approach for adapting LLMs to target tasks. However, previous methods are limited in adaptability as they keep the LoRA parameters frozen. Additionally, the low-resource problem has been out of their scope. We propose a LoRA merge method that updates and prunes LoRA parameters through fine-tuning with minimal target task data, which allows finer-grained adjustments of LoRA parameters and enhancement of task adaptability. Extensive experiments have been conducted taking summarization as a benchmark task. Our datasets cover various domains and multiple languages of English and Japanese. The results confirm that the proposed method achieves significant and consistent improvements in task adaptability over the previous methods. 

---
# Tag-Evol: Achieving Efficient Instruction Evolving via Tag Injection 

**Authors**: Yixuan Wang, Shiqi Zhou, Chuanzhe Guo, Qingfu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24165)  

**Abstract**: Evol-Instruct has made significant improvements as a data synthesis method in several areas. Existing methods typically rely on a fixed set of strategies to evolve, which require manual design and are monolithic in form. In addition, iterative evolution also makes the acquisition of hard samples expensive. In view of this, we propose the Tag-Evol framework, a more diverse and efficient instruction evolving method. Specifically, Tag-Evol uses diverse and specific knowledge tags as strategies to achieve controlled evolution by injecting different combinations of tags into the original instructions. Experiments with multiple backbones in diverse domain benchmarks show that the proposed method generates significantly better evolved data than other methods. Furthermore, we conduct a thorough analysis of the evolved data, demonstrating that Tag-Evol is not only efficient but also generates more diverse and challenging data. 

---
# Mixed-R1: Unified Reward Perspective For Reasoning Capability in Multimodal Large Language Models 

**Authors**: Shilin Xu, Yanwei Li, Rui Yang, Tao Zhang, Yueyi Sun, Wei Chow, Linfeng Li, Hang Song, Qi Xu, Yunhai Tong, Xiangtai Li, Hao Fei  

**Link**: [PDF](https://arxiv.org/pdf/2505.24164)  

**Abstract**: Recent works on large language models (LLMs) have successfully demonstrated the emergence of reasoning capabilities via reinforcement learning (RL). Although recent efforts leverage group relative policy optimization (GRPO) for MLLMs post-training, they constantly explore one specific aspect, such as grounding tasks, math problems, or chart analysis. There are no works that can leverage multi-source MLLM tasks for stable reinforcement learning. In this work, we present a unified perspective to solve this problem. We present Mixed-R1, a unified yet straightforward framework that contains a mixed reward function design (Mixed-Reward) and a mixed post-training dataset (Mixed-45K). We first design a data engine to select high-quality examples to build the Mixed-45K post-training dataset. Then, we present a Mixed-Reward design, which contains various reward functions for various MLLM tasks. In particular, it has four different reward functions: matching reward for binary answer or multiple-choice problems, chart reward for chart-aware datasets, IoU reward for grounding problems, and open-ended reward for long-form text responses such as caption datasets. To handle the various long-form text content, we propose a new open-ended reward named Bidirectional Max-Average Similarity (BMAS) by leveraging tokenizer embedding matching between the generated response and the ground truth. Extensive experiments show the effectiveness of our proposed method on various MLLMs, including Qwen2.5-VL and Intern-VL on various sizes. Our dataset and model are available at this https URL. 

---
# LKD-KGC: Domain-Specific KG Construction via LLM-driven Knowledge Dependency Parsing 

**Authors**: Jiaqi Sun, Shiyou Qian, Zhangchi Han, Wei Li, Zelin Qian, Dingyu Yang, Jian Cao, Guangtao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2505.24163)  

**Abstract**: Knowledge Graphs (KGs) structure real-world entities and their relationships into triples, enhancing machine reasoning for various tasks. While domain-specific KGs offer substantial benefits, their manual construction is often inefficient and requires specialized knowledge. Recent approaches for knowledge graph construction (KGC) based on large language models (LLMs), such as schema-guided KGC and reference knowledge integration, have proven efficient. However, these methods are constrained by their reliance on manually defined schema, single-document processing, and public-domain references, making them less effective for domain-specific corpora that exhibit complex knowledge dependencies and specificity, as well as limited reference knowledge. To address these challenges, we propose LKD-KGC, a novel framework for unsupervised domain-specific KG construction. LKD-KGC autonomously analyzes document repositories to infer knowledge dependencies, determines optimal processing sequences via LLM driven prioritization, and autoregressively generates entity schema by integrating hierarchical inter-document contexts. This schema guides the unsupervised extraction of entities and relationships, eliminating reliance on predefined structures or external knowledge. Extensive experiments show that compared with state-of-the-art baselines, LKD-KGC generally achieves improvements of 10% to 20% in both precision and recall rate, demonstrating its potential in constructing high-quality domain-specific KGs. 

---
# Rationales Are Not Silver Bullets: Measuring the Impact of Rationales on Model Performance and Reliability 

**Authors**: Chiwei Zhu, Benfeng Xu, An Yang, Junyang Lin, Quan Wang, Chang Zhou, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.24147)  

**Abstract**: Training language models with rationales augmentation has been shown to be beneficial in many existing works. In this paper, we identify that such a prevailing view does not hold consistently. We conduct comprehensive investigations to thoroughly inspect the impact of rationales on model performance as well as a novel perspective of model reliability. The results lead to several key findings that add new insights upon existing understandings: 1) Rationales can, at times, deteriorate model performance; 2) Rationales can, at times, improve model reliability, even outperforming their untrained counterparts; 3) A linear correspondence exists in between the performance and reliability improvements, while both are driven by the intrinsic difficulty of the task. These findings provide informative regulations on the broad utilization of rationales and raise critical implications on the procedure of explicitly aligning language models with implicit human thoughts. Codes can be found at this https URL. 

---
# CrossICL: Cross-Task In-Context Learning via Unsupervised Demonstration Transfer 

**Authors**: Jinglong Gao, Xiao Ding, Lingxiao Zou, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24143)  

**Abstract**: In-Context Learning (ICL) enhances the performance of large language models (LLMs) with demonstrations. However, obtaining these demonstrations primarily relies on manual effort. In most real-world scenarios, users are often unwilling or unable to provide such demonstrations. Inspired by the human analogy, we explore a new ICL paradigm CrossICL to study how to utilize existing source task demonstrations in the ICL for target tasks, thereby obtaining reliable guidance without any additional manual effort. To explore this, we first design a two-stage alignment strategy to mitigate the interference caused by gaps across tasks, as the foundation for our experimental exploration. Based on it, we conduct comprehensive exploration of CrossICL, with 875 NLP tasks from the Super-NI benchmark and six types of LLMs, including GPT-4o. Experimental results demonstrate the effectiveness of CrossICL and provide valuable insights on questions like the criteria for selecting cross-task demonstrations, as well as the types of task-gap-induced interference in CrossICL. 

---
# R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration 

**Authors**: Zefan Cai, Wen Xiao, Hanshi Sun, Cheng Luo, Yikai Zhang, Ke Wan, Yucheng Li, Yeyang Zhou, Li-Wen Chang, Jiuxiang Gu, Zhen Dong, Anima Anandkumar, Abedelkadir Asi, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24133)  

**Abstract**: Reasoning models have demonstrated impressive performance in self-reflection and chain-of-thought reasoning. However, they often produce excessively long outputs, leading to prohibitively large key-value (KV) caches during inference. While chain-of-thought inference significantly improves performance on complex reasoning tasks, it can also lead to reasoning failures when deployed with existing KV cache compression approaches. To address this, we propose Redundancy-aware KV Cache Compression for Reasoning models (R-KV), a novel method specifically targeting redundant tokens in reasoning models. Our method preserves nearly 100% of the full KV cache performance using only 10% of the KV cache, substantially outperforming existing KV cache baselines, which reach only 60% of the performance. Remarkably, R-KV even achieves 105% of full KV cache performance with 16% of the KV cache. This KV-cache reduction also leads to a 90% memory saving and a 6.6X throughput over standard chain-of-thought reasoning inference. Experimental results show that R-KV consistently outperforms existing KV cache compression baselines across two mathematical reasoning datasets. 

---
# The State of Multilingual LLM Safety Research: From Measuring the Language Gap to Mitigating It 

**Authors**: Zheng-Xin Yong, Beyza Ermis, Marzieh Fadaee, Stephen H. Bach, Julia Kreutzer  

**Link**: [PDF](https://arxiv.org/pdf/2505.24119)  

**Abstract**: This paper presents a comprehensive analysis of the linguistic diversity of LLM safety research, highlighting the English-centric nature of the field. Through a systematic review of nearly 300 publications from 2020--2024 across major NLP conferences and workshops at *ACL, we identify a significant and growing language gap in LLM safety research, with even high-resource non-English languages receiving minimal attention. We further observe that non-English languages are rarely studied as a standalone language and that English safety research exhibits poor language documentation practice. To motivate future research into multilingual safety, we make several recommendations based on our survey, and we then pose three concrete future directions on safety evaluation, training data generation, and crosslingual safety generalization. Based on our survey and proposed directions, the field can develop more robust, inclusive AI safety practices for diverse global populations. 

---
# Training LLMs for EHR-Based Reasoning Tasks via Reinforcement Learning 

**Authors**: Jiacheng Lin, Zhenbang Wu, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.24105)  

**Abstract**: We present EHRMIND, a practical recipe for adapting large language models (LLMs) to complex clinical reasoning tasks using reinforcement learning with verifiable rewards (RLVR). While RLVR has succeeded in mathematics and coding, its application to healthcare contexts presents unique challenges due to the specialized knowledge and reasoning required for electronic health record (EHR) interpretation. Our pilot study on the MEDCALC benchmark reveals two key failure modes: (1) misapplied knowledge, where models possess relevant medical knowledge but apply it incorrectly, and (2) missing knowledge, where models lack essential domain knowledge. To address these cases, EHRMIND applies a two-stage solution: a lightweight supervised fine-tuning (SFT) warm-up that injects missing domain knowledge, stabilizes subsequent training, and encourages structured, interpretable outputs; followed by RLVR, which reinforces outcome correctness and refines the model's decision-making. We demonstrate the effectiveness of our method across diverse clinical applications, including medical calculations (MEDCALC), patient-trial matching (TREC CLINICAL TRIALS), and disease diagnosis (EHRSHOT). EHRMIND delivers consistent gains in accuracy, interpretability, and cross-task generalization. These findings offer practical guidance for applying RLVR to enhance LLM capabilities in healthcare settings. 

---
# HardTests: Synthesizing High-Quality Test Cases for LLM Coding 

**Authors**: Zhongmou He, Yee Man Choi, Kexun Zhang, Jiabao Ji, Junting Zhou, Dejia Xu, Ivan Bercovich, Aidan Zhang, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24098)  

**Abstract**: Verifiers play a crucial role in large language model (LLM) reasoning, needed by post-training techniques such as reinforcement learning. However, reliable verifiers are hard to get for difficult coding problems, because a well-disguised wrong solution may only be detected by carefully human-written edge cases that are difficult to synthesize. To address this issue, we propose HARDTESTGEN, a pipeline for high-quality test synthesis using LLMs. With this pipeline, we curate a comprehensive competitive programming dataset HARDTESTS with 47k problems and synthetic high-quality tests. Compared with existing tests, HARDTESTGEN tests demonstrate precision that is 11.3 percentage points higher and recall that is 17.5 percentage points higher when evaluating LLM-generated code. For harder problems, the improvement in precision can be as large as 40 points. HARDTESTS also proves to be more effective for model training, measured by downstream code generation performance. We will open-source our dataset and synthesis pipeline at this https URL. 

---
# TCM-Ladder: A Benchmark for Multimodal Question Answering on Traditional Chinese Medicine 

**Authors**: Jiacheng Xie, Yang Yu, Ziyang Zhang, Shuai Zeng, Jiaxuan He, Ayush Vasireddy, Xiaoting Tang, Congyu Guo, Lening Zhao, Congcong Jing, Guanghui An, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24063)  

**Abstract**: Traditional Chinese Medicine (TCM), as an effective alternative medicine, has been receiving increasing attention. In recent years, the rapid development of large language models (LLMs) tailored for TCM has underscored the need for an objective and comprehensive evaluation framework to assess their performance on real-world tasks. However, existing evaluation datasets are limited in scope and primarily text-based, lacking a unified and standardized multimodal question-answering (QA) benchmark. To address this issue, we introduce TCM-Ladder, the first multimodal QA dataset specifically designed for evaluating large TCM language models. The dataset spans multiple core disciplines of TCM, including fundamental theory, diagnostics, herbal formulas, internal medicine, surgery, pharmacognosy, and pediatrics. In addition to textual content, TCM-Ladder incorporates various modalities such as images and videos. The datasets were constructed using a combination of automated and manual filtering processes and comprise 52,000+ questions in total. These questions include single-choice, multiple-choice, fill-in-the-blank, diagnostic dialogue, and visual comprehension tasks. We trained a reasoning model on TCM-Ladder and conducted comparative experiments against 9 state-of-the-art general domain and 5 leading TCM-specific LLMs to evaluate their performance on the datasets. Moreover, we propose Ladder-Score, an evaluation method specifically designed for TCM question answering that effectively assesses answer quality regarding terminology usage and semantic expression. To our knowledge, this is the first work to evaluate mainstream general domain and TCM-specific LLMs on a unified multimodal benchmark. The datasets and leaderboard are publicly available at this https URL or this https URL and will be continuously updated. 

---
# MedPAIR: Measuring Physicians and AI Relevance Alignment in Medical Question Answering 

**Authors**: Yuexing Hao, Kumail Alhamoud, Hyewon Jeong, Haoran Zhang, Isha Puri, Philip Torr, Mike Schaekermann, Ariel D. Stern, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2505.24040)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance on various medical question-answering (QA) benchmarks, including standardized medical exams. However, correct answers alone do not ensure correct logic, and models may reach accurate conclusions through flawed processes. In this study, we introduce the MedPAIR (Medical Dataset Comparing Physicians and AI Relevance Estimation and Question Answering) dataset to evaluate how physician trainees and LLMs prioritize relevant information when answering QA questions. We obtain annotations on 1,300 QA pairs from 36 physician trainees, labeling each sentence within the question components for relevance. We compare these relevance estimates to those for LLMs, and further evaluate the impact of these "relevant" subsets on downstream task performance for both physician trainees and LLMs. We find that LLMs are frequently not aligned with the content relevance estimates of physician trainees. After filtering out physician trainee-labeled irrelevant sentences, accuracy improves for both the trainees and the LLMs. All LLM and physician trainee-labeled data are available at: this http URL. 

---
# The Surprising Soupability of Documents in State Space Models 

**Authors**: Yasaman Jafari, Zixian Wang, Leon Bergen, Taylor Berg-Kirkpatrick  

**Link**: [PDF](https://arxiv.org/pdf/2505.24033)  

**Abstract**: We investigate whether hidden states from Structured State Space Models (SSMs) can be merged post-hoc to support downstream reasoning. Inspired by model souping, we propose a strategy where documents are encoded independently and their representations are pooled -- via simple operations like averaging -- into a single context state. This approach, which we call document souping, enables modular encoding and reuse without reprocessing the full input for each query. We finetune Mamba2 models to produce soupable representations and find that they support multi-hop QA, sparse retrieval, and long-document reasoning with strong accuracy. On HotpotQA, souping ten independently encoded documents nearly matches the performance of a cross-encoder trained on the same inputs. 

---
# Hidden Persuasion: Detecting Manipulative Narratives on Social Media During the 2022 Russian Invasion of Ukraine 

**Authors**: Kateryna Akhynko, Oleksandr Kosovan, Mykola Trokhymovych  

**Link**: [PDF](https://arxiv.org/pdf/2505.24028)  

**Abstract**: This paper presents one of the top-performing solutions to the UNLP 2025 Shared Task on Detecting Manipulation in Social Media. The task focuses on detecting and classifying rhetorical and stylistic manipulation techniques used to influence Ukrainian Telegram users. For the classification subtask, we fine-tuned the Gemma 2 language model with LoRA adapters and applied a second-level classifier leveraging meta-features and threshold optimization. For span detection, we employed an XLM-RoBERTa model trained for multi-target, including token binary classification. Our approach achieved 2nd place in classification and 3rd place in span detection. 

---
# BeaverTalk: Oregon State University's IWSLT 2025 Simultaneous Speech Translation System 

**Authors**: Matthew Raffel, Victor Agostinelli, Lizhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24016)  

**Abstract**: This paper discusses the construction, fine-tuning, and deployment of BeaverTalk, a cascaded system for speech-to-text translation as part of the IWSLT 2025 simultaneous translation task. The system architecture employs a VAD segmenter for breaking a speech stream into segments, Whisper Large V2 for automatic speech recognition (ASR), and Gemma 3 12B for simultaneous translation. Regarding the simultaneous translation LLM, it is fine-tuned via low-rank adaptors (LoRAs) for a conversational prompting strategy that leverages a single prior-sentence memory bank from the source language as context. The cascaded system participated in the English$\rightarrow$German and English$\rightarrow$Chinese language directions for both the low and high latency regimes. In particular, on the English$\rightarrow$German task, the system achieves a BLEU of 24.64 and 27.83 at a StreamLAAL of 1837.86 and 3343.73, respectively. Then, on the English$\rightarrow$Chinese task, the system achieves a BLEU of 34.07 and 37.23 at a StreamLAAL of 2216.99 and 3521.35, respectively. 

---
# Large Language Model Meets Constraint Propagation 

**Authors**: Alexandre Bonlarron, Florian Régin, Elisabetta De Maria, Jean-Charles Régin  

**Link**: [PDF](https://arxiv.org/pdf/2505.24012)  

**Abstract**: Large Language Models (LLMs) excel at generating fluent text but struggle to enforce external constraints because they generate tokens sequentially without explicit control mechanisms. GenCP addresses this limitation by combining LLM predictions with Constraint Programming (CP) reasoning, formulating text generation as a Constraint Satisfaction Problem (CSP). In this paper, we improve GenCP by integrating Masked Language Models (MLMs) for domain generation, which allows bidirectional constraint propagation that leverages both past and future tokens. This integration bridges the gap between token-level prediction and structured constraint enforcement, leading to more reliable and constraint-aware text generation. Our evaluation on COLLIE benchmarks demonstrates that incorporating domain preview via MLM calls significantly improves GenCP's performance. Although this approach incurs additional MLM calls and, in some cases, increased backtracking, the overall effect is a more efficient use of LLM inferences and an enhanced ability to generate feasible and meaningful solutions, particularly in tasks with strict content constraints. 

---
# Diversity of Transformer Layers: One Aspect of Parameter Scaling Laws 

**Authors**: Hidetaka Kamigaito, Ying Zhang, Jingun Kwon, Katsuhiko Hayashi, Manabu Okumura, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2505.24009)  

**Abstract**: Transformers deliver outstanding performance across a wide range of tasks and are now a dominant backbone architecture for large language models (LLMs). Their task-solving performance is improved by increasing parameter size, as shown in the recent studies on parameter scaling laws. Although recent mechanistic-interpretability studies have deepened our understanding of the internal behavior of Transformers by analyzing their residual stream, the relationship between these internal mechanisms and the parameter scaling laws remains unclear. To bridge this gap, we focus on layers and their size, which mainly decide the parameter size of Transformers. For this purpose, we first theoretically investigate the layers within the residual stream through a bias-diversity decomposition. The decomposition separates (i) bias, the error of each layer's output from the ground truth, and (ii) diversity, which indicates how much the outputs of each layer differ from each other. Analyzing Transformers under this theory reveals that performance improves when individual layers make predictions close to the correct answer and remain mutually diverse. We show that diversity becomes especially critical when individual layers' outputs are far from the ground truth. Finally, we introduce an information-theoretic diversity and show our main findings that adding layers enhances performance only when those layers behave differently, i.e., are diverse. We also reveal the performance gains from increasing the number of layers exhibit submodularity: marginal improvements diminish as additional layers increase, mirroring the logarithmic convergence predicted by the parameter scaling laws. Experiments on multiple semantic-understanding tasks with various LLMs empirically confirm the theoretical properties derived in this study. 

---
# Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs 

**Authors**: Yinong Oliver Wang, Nivedha Sivakumar, Falaah Arif Khan, Rin Metcalf Susa, Adam Golinski, Natalie Mackraz, Barry-John Theobald, Luca Zappella, Nicholas Apostoloff  

**Link**: [PDF](https://arxiv.org/pdf/2505.23996)  

**Abstract**: The recent rapid adoption of large language models (LLMs) highlights the critical need for benchmarking their fairness. Conventional fairness metrics, which focus on discrete accuracy-based evaluations (i.e., prediction correctness), fail to capture the implicit impact of model uncertainty (e.g., higher model confidence about one group over another despite similar accuracy). To address this limitation, we propose an uncertainty-aware fairness metric, UCerF, to enable a fine-grained evaluation of model fairness that is more reflective of the internal bias in model decisions compared to conventional fairness measures. Furthermore, observing data size, diversity, and clarity issues in current datasets, we introduce a new gender-occupation fairness evaluation dataset with 31,756 samples for co-reference resolution, offering a more diverse and suitable dataset for evaluating modern LLMs. We establish a benchmark, using our metric and dataset, and apply it to evaluate the behavior of ten open-source LLMs. For example, Mistral-7B exhibits suboptimal fairness due to high confidence in incorrect predictions, a detail overlooked by Equalized Odds but captured by UCerF. Overall, our proposed LLM benchmark, which evaluates fairness with uncertainty awareness, paves the way for developing more transparent and accountable AI systems. 

---
# FLAT-LLM: Fine-grained Low-rank Activation Space Transformation for Large Language Model Compression 

**Authors**: Jiayi Tian, Ryan Solgi, Jinming Lu, Yifan Yang, Hai Li, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23966)  

**Abstract**: Large Language Models (LLMs) have enabled remarkable progress in natural language processing, yet their high computational and memory demands pose challenges for deployment in resource-constrained environments. Although recent low-rank decomposition methods offer a promising path for structural compression, they often suffer from accuracy degradation, expensive calibration procedures, and result in inefficient model architectures that hinder real-world inference speedups. In this paper, we propose FLAT-LLM, a fast and accurate, training-free structural compression method based on fine-grained low-rank transformations in the activation space. Specifically, we reduce the hidden dimension by transforming the weights using truncated eigenvectors computed via head-wise Principal Component Analysis (PCA), and employ an importance-based metric to adaptively allocate ranks across decoders. FLAT-LLM achieves efficient and effective weight compression without recovery fine-tuning, which could complete the calibration within a few minutes. Evaluated across 4 models and 11 datasets, FLAT-LLM outperforms structural pruning baselines in generalization and downstream performance, while delivering inference speedups over decomposition-based methods. 

---
# A Closer Look at Bias and Chain-of-Thought Faithfulness of Large (Vision) Language Models 

**Authors**: Sriram Balasubramanian, Samyadeep Basu, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2505.23945)  

**Abstract**: Chain-of-thought (CoT) reasoning enhances performance of large language models, but questions remain about whether these reasoning traces faithfully reflect the internal processes of the model. We present the first comprehensive study of CoT faithfulness in large vision-language models (LVLMs), investigating how both text-based and previously unexplored image-based biases affect reasoning and bias articulation. Our work introduces a novel, fine-grained evaluation pipeline for categorizing bias articulation patterns, enabling significantly more precise analysis of CoT reasoning than previous methods. This framework reveals critical distinctions in how models process and respond to different types of biases, providing new insights into LVLM CoT faithfulness. Our findings reveal that subtle image-based biases are rarely articulated compared to explicit text-based ones, even in models specialized for reasoning. Additionally, many models exhibit a previously unidentified phenomenon we term ``inconsistent'' reasoning - correctly reasoning before abruptly changing answers, serving as a potential canary for detecting biased reasoning from unfaithful CoTs. We then apply the same evaluation pipeline to revisit CoT faithfulness in LLMs across various levels of implicit cues. Our findings reveal that current language-only reasoning models continue to struggle with articulating cues that are not overtly stated. 

---
# Retrieval Augmented Generation based Large Language Models for Causality Mining 

**Authors**: Thushara Manjari Naduvilakandy, Hyeju Jang, Mohammad Al Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2505.23944)  

**Abstract**: Causality detection and mining are important tasks in information retrieval due to their enormous use in information extraction, and knowledge graph construction. To solve these tasks, in existing literature there exist several solutions -- both unsupervised and supervised. However, the unsupervised methods suffer from poor performance and they often require significant human intervention for causal rule selection, leading to poor generalization across different domains. On the other hand, supervised methods suffer from the lack of large training datasets. Recently, large language models (LLMs) with effective prompt engineering are found to be effective to overcome the issue of unavailability of large training dataset. Yet, in existing literature, there does not exist comprehensive works on causality detection and mining using LLM prompting. In this paper, we present several retrieval-augmented generation (RAG) based dynamic prompting schemes to enhance LLM performance in causality detection and extraction tasks. Extensive experiments over three datasets and five LLMs validate the superiority of our proposed RAG-based dynamic prompting over other static prompting schemes. 

---
# SwingArena: Competitive Programming Arena for Long-context GitHub Issue Solving 

**Authors**: Wendong Xu, Jing Xiong, Chenyang Zhao, Qiujiang Chen, Haoran Wang, Hui Shen, Zhongwei Wan, Jianbo Dai, Taiqiang Wu, He Xiao, Chaofan Tao, Z. Morley Mao, Ying Sheng, Zhijiang Guo, Hongxia Yang, Bei Yu, Lingpeng Kong, Quanquan Gu, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2505.23932)  

**Abstract**: We present SwingArena, a competitive evaluation framework for Large Language Models (LLMs) that closely mirrors real-world software development workflows. Unlike traditional static benchmarks, SwingArena models the collaborative process of software iteration by pairing LLMs as submitters, who generate patches, and reviewers, who create test cases and verify the patches through continuous integration (CI) pipelines. To support these interactive evaluations, we introduce a retrieval-augmented code generation (RACG) module that efficiently handles long-context challenges by providing syntactically and semantically relevant code snippets from large codebases, supporting multiple programming languages (C++, Python, Rust, and Go). This enables the framework to scale across diverse tasks and contexts while respecting token limitations. Our experiments, using over 400 high-quality real-world GitHub issues selected from a pool of 2,300 issues, show that models like GPT-4o excel at aggressive patch generation, whereas DeepSeek and Gemini prioritize correctness in CI validation. SwingArena presents a scalable and extensible methodology for evaluating LLMs in realistic, CI-driven software development settings. More details are available on our project page: this http URL 

---
# Scaling up the think-aloud method 

**Authors**: Daniel Wurgaft, Ben Prystawski, Kanishk Gandhi, Cedegao E. Zhang, Joshua B. Tenenbaum, Noah D. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2505.23931)  

**Abstract**: The think-aloud method, where participants voice their thoughts as they solve a task, is a valuable source of rich data about human reasoning processes. Yet, it has declined in popularity in contemporary cognitive science, largely because labor-intensive transcription and annotation preclude large sample sizes. Here, we develop methods to automate the transcription and annotation of verbal reports of reasoning using natural language processing tools, allowing for large-scale analysis of think-aloud data. In our study, 640 participants thought aloud while playing the Game of 24, a mathematical reasoning task. We automatically transcribed the recordings and coded the transcripts as search graphs, finding moderate inter-rater reliability with humans. We analyze these graphs and characterize consistency and variation in human reasoning traces. Our work demonstrates the value of think-aloud data at scale and serves as a proof of concept for the automated analysis of verbal reports. 

---
# ChARM: Character-based Act-adaptive Reward Modeling for Advanced Role-Playing Language Agents 

**Authors**: Feiteng Fang, Ting-En Lin, Yuchuan Wu, Xiong Liu, Xiang Huang, Dingwei Chen, Jing Ye, Haonan Zhang, Liang Zhu, Hamid Alinejad-Rokny, Min Yang, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23923)  

**Abstract**: Role-Playing Language Agents (RPLAs) aim to simulate characters for realistic and engaging human-computer interactions. However, traditional reward models often struggle with scalability and adapting to subjective conversational preferences. We propose ChARM, a Character-based Act-adaptive Reward Model, addressing these challenges through two innovations: (1) an act-adaptive margin that significantly enhances learning efficiency and generalizability, and (2) a self-evolution mechanism leveraging large-scale unlabeled data to improve training coverage. Additionally, we introduce RoleplayPref, the first large-scale preference dataset specifically for RPLAs, featuring 1,108 characters, 13 subcategories, and 16,888 bilingual dialogues, alongside RoleplayEval, a dedicated evaluation benchmark. Experimental results show a 13% improvement over the conventional Bradley-Terry model in preference rankings. Furthermore, applying ChARM-generated rewards to preference learning techniques (e.g., direct preference optimization) achieves state-of-the-art results on CharacterEval and RoleplayEval. Code and dataset are available at this https URL. 

---
# Probing Association Biases in LLM Moderation Over-Sensitivity 

**Authors**: Yuxin Wang, Botao Yu, Ivory Yang, Saeed Hassanpour, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2505.23914)  

**Abstract**: Large Language Models are widely used for content moderation but often misclassify benign comments as toxic, leading to over-sensitivity. While previous research attributes this issue primarily to the presence of offensive terms, we reveal a potential cause beyond token level: LLMs exhibit systematic topic biases in their implicit associations. Inspired by cognitive psychology's implicit association tests, we introduce Topic Association Analysis, a semantic-level approach to quantify how LLMs associate certain topics with toxicity. By prompting LLMs to generate free-form scenario imagination for misclassified benign comments and analyzing their topic amplification levels, we find that more advanced models (e.g., GPT-4 Turbo) demonstrate stronger topic stereotype despite lower overall false positive rates. These biases suggest that LLMs do not merely react to explicit, offensive language but rely on learned topic associations, shaping their moderation decisions. Our findings highlight the need for refinement beyond keyword-based filtering, providing insights into the underlying mechanisms driving LLM over-sensitivity. 

---
# Reinforcement Learning for Better Verbalized Confidence in Long-Form Generation 

**Authors**: Caiqi Zhang, Xiaochen Zhu, Chengzu Li, Nigel Collier, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2505.23912)  

**Abstract**: Hallucination remains a major challenge for the safe and trustworthy deployment of large language models (LLMs) in factual content generation. Prior work has explored confidence estimation as an effective approach to hallucination detection, but often relies on post-hoc self-consistency methods that require computationally expensive sampling. Verbalized confidence offers a more efficient alternative, but existing approaches are largely limited to short-form question answering (QA) tasks and do not generalize well to open-ended generation. In this paper, we propose LoVeC (Long-form Verbalized Confidence), an on-the-fly verbalized confidence estimation method for long-form generation. Specifically, we use reinforcement learning (RL) to train LLMs to append numerical confidence scores to each generated statement, serving as a direct and interpretable signal of the factuality of generation. Our experiments consider both on-policy and off-policy RL methods, including DPO, ORPO, and GRPO, to enhance the model calibration. We introduce two novel evaluation settings, free-form tagging and iterative tagging, to assess different verbalized confidence estimation methods. Experiments on three long-form QA datasets show that our RL-trained models achieve better calibration and generalize robustly across domains. Also, our method is highly efficient, as it only requires adding a few tokens to the output being decoded. 

---
# One Task Vector is not Enough: A Large-Scale Study for In-Context Learning 

**Authors**: Pavel Tikhonov, Ivan Oseledets, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2505.23911)  

**Abstract**: In-context learning (ICL) enables Large Language Models (LLMs) to adapt to new tasks using few examples, with task vectors - specific hidden state activations - hypothesized to encode task information. Existing studies are limited by small-scale benchmarks, restricting comprehensive analysis. We introduce QuiteAFew, a novel dataset of 3,096 diverse few-shot tasks, each with 30 input-output pairs derived from the Alpaca dataset. Experiments with Llama-3-8B on QuiteAFew reveal: (1) task vector performance peaks at an intermediate layer (e.g., 15th), (2) effectiveness varies significantly by task type, and (3) complex tasks rely on multiple, subtask-specific vectors rather than a single vector, suggesting distributed task knowledge representation. 

---
# Infi-Med: Low-Resource Medical MLLMs with Robust Reasoning Evaluation 

**Authors**: Zeyu Liu, Zhitian Hou, Yining Di, Kejing Yang, Zhijie Sang, Congkai Xie, Jingwen Yang, Siyuan Liu, Jialu Wang, Chunming Li, Ming Li, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23867)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated promising prospects in healthcare, particularly for addressing complex medical tasks, supporting multidisciplinary treatment (MDT), and enabling personalized precision medicine. However, their practical deployment faces critical challenges in resource efficiency, diagnostic accuracy, clinical considerations, and ethical privacy. To address these limitations, we propose Infi-Med, a comprehensive framework for medical MLLMs that introduces three key innovations: (1) a resource-efficient approach through curating and constructing high-quality supervised fine-tuning (SFT) datasets with minimal sample requirements, with a forward-looking design that extends to both pretraining and posttraining phases; (2) enhanced multimodal reasoning capabilities for cross-modal integration and clinical task understanding; and (3) a systematic evaluation system that assesses model performance across medical modalities and task types. Our experiments demonstrate that Infi-Med achieves state-of-the-art (SOTA) performance in general medical reasoning while maintaining rapid adaptability to clinical scenarios. The framework establishes a solid foundation for deploying MLLMs in real-world healthcare settings by balancing model effectiveness with operational constraints. 

---
# OMNIGUARD: An Efficient Approach for AI Safety Moderation Across Modalities 

**Authors**: Sahil Verma, Keegan Hines, Jeff Bilmes, Charlotte Siska, Luke Zettlemoyer, Hila Gonen, Chandan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.23856)  

**Abstract**: The emerging capabilities of large language models (LLMs) have sparked concerns about their immediate potential for harmful misuse. The core approach to mitigate these concerns is the detection of harmful queries to the model. Current detection approaches are fallible, and are particularly susceptible to attacks that exploit mismatched generalization of model capabilities (e.g., prompts in low-resource languages or prompts provided in non-text modalities such as image and audio). To tackle this challenge, we propose OMNIGUARD, an approach for detecting harmful prompts across languages and modalities. Our approach (i) identifies internal representations of an LLM/MLLM that are aligned across languages or modalities and then (ii) uses them to build a language-agnostic or modality-agnostic classifier for detecting harmful prompts. OMNIGUARD improves harmful prompt classification accuracy by 11.57\% over the strongest baseline in a multilingual setting, by 20.44\% for image-based prompts, and sets a new SOTA for audio-based prompts. By repurposing embeddings computed during generation, OMNIGUARD is also very efficient ($\approx 120 \times$ faster than the next fastest baseline). Code and data are available at: this https URL. 

---
# Revisiting Uncertainty Estimation and Calibration of Large Language Models 

**Authors**: Linwei Tao, Yi-Fan Yeh, Minjing Dong, Tao Huang, Philip Torr, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23854)  

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes applications, robust uncertainty estimation is essential for ensuring the safe and trustworthy deployment of LLMs. We present the most comprehensive study to date of uncertainty estimation in LLMs, evaluating 80 models spanning open- and closed-source families, dense and Mixture-of-Experts (MoE) architectures, reasoning and non-reasoning modes, quantization variants and parameter scales from 0.6B to 671B. Focusing on three representative black-box single-pass methods, including token probability-based uncertainty (TPU), numerical verbal uncertainty (NVU), and linguistic verbal uncertainty (LVU), we systematically evaluate uncertainty calibration and selective classification using the challenging MMLU-Pro benchmark, which covers both reasoning-intensive and knowledge-based tasks. Our results show that LVU consistently outperforms TPU and NVU, offering stronger calibration and discrimination while being more interpretable. We also find that high accuracy does not imply reliable uncertainty, and that model scale, post-training, reasoning ability and quantization all influence estimation performance. Notably, LLMs exhibit better uncertainty estimates on reasoning tasks than on knowledge-heavy ones, and good calibration does not necessarily translate to effective error ranking. These findings highlight the need for multi-perspective evaluation and position LVU as a practical tool for improving the reliability of LLMs in real-world settings. 

---
# Large Language Model-Based Agents for Automated Research Reproducibility: An Exploratory Study in Alzheimer's Disease 

**Authors**: Nic Dobbins, Christelle Xiong, Kristine Lan, Meliha Yetisgen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23852)  

**Abstract**: Objective: To demonstrate the capabilities of Large Language Models (LLMs) as autonomous agents to reproduce findings of published research studies using the same or similar dataset.
Materials and Methods: We used the "Quick Access" dataset of the National Alzheimer's Coordinating Center (NACC). We identified highly cited published research manuscripts using NACC data and selected five studies that appeared reproducible using this dataset alone. Using GPT-4o, we created a simulated research team of LLM-based autonomous agents tasked with writing and executing code to dynamically reproduce the findings of each study, given only study Abstracts, Methods sections, and data dictionary descriptions of the dataset.
Results: We extracted 35 key findings described in the Abstracts across 5 Alzheimer's studies. On average, LLM agents approximately reproduced 53.2% of findings per study. Numeric values and range-based findings often differed between studies and agents. The agents also applied statistical methods or parameters that varied from the originals, though overall trends and significance were sometimes similar.
Discussion: In some cases, LLM-based agents replicated research techniques and findings. In others, they failed due to implementation flaws or missing methodological detail. These discrepancies show the current limits of LLMs in fully automating reproducibility assessments. Still, this early investigation highlights the potential of structured agent-based systems to provide scalable evaluation of scientific rigor.
Conclusion: This exploratory work illustrates both the promise and limitations of LLMs as autonomous agents for automating reproducibility in biomedical research. 

---
# ASyMOB: Algebraic Symbolic Mathematical Operations Benchmark 

**Authors**: Michael Shalyt, Rotem Elimelech, Ido Kaminer  

**Link**: [PDF](https://arxiv.org/pdf/2505.23851)  

**Abstract**: Large language models (LLMs) are rapidly approaching the level of proficiency in university-level symbolic mathematics required for applications in advanced science and technology. However, existing benchmarks fall short in assessing the core skills of LLMs in symbolic mathematics-such as integration, differential equations, and algebraic simplification. To address this gap, we introduce ASyMOB, a novel assessment framework focused exclusively on symbolic manipulation, featuring 17,092 unique math challenges, organized by similarity and complexity. ASyMOB enables analysis of LLM generalization capabilities by comparing performance in problems that differ by simple numerical or symbolic `perturbations'. Evaluated LLMs exhibit substantial degradation in performance for all perturbation types (up to -70.3%), suggesting reliance on memorized patterns rather than deeper understanding of symbolic math, even among models achieving high baseline accuracy. Comparing LLM performance to computer algebra systems, we identify examples where they fail while LLMs succeed, as well as problems solved only by combining both approaches. Models capable of integrated code execution yielded higher accuracy compared to their performance without code, particularly stabilizing weaker models (up to +33.1% for certain perturbation types). Notably, the most advanced models (o4-mini, Gemini 2.5 Flash) demonstrate not only high symbolic math proficiency (scoring 96.8% and 97.6% on the unperturbed set), but also remarkable robustness against perturbations, (-21.7% and -21.2% vs. average -50.4% for the other models). This may indicate a recent "phase transition" in the generalization capabilities of frontier LLMs. It remains to be seen whether the path forward lies in deeper integration with sophisticated external tools, or in developing models so capable that symbolic math systems like CAS become unnecessary. 

---
# Derailing Non-Answers via Logit Suppression at Output Subspace Boundaries in RLHF-Aligned Language Models 

**Authors**: Harvey Dam, Jonas Knochelmann, Vinu Joseph, Ganesh Gopalakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.23848)  

**Abstract**: We introduce a method to reduce refusal rates of large language models (LLMs) on sensitive content without modifying model weights or prompts. Motivated by the observation that refusals in certain models were often preceded by the specific token sequence of a token marking the beginning of the chain-of-thought (CoT) block (<think>) followed by a double newline token (\n\n), we investigate the impact of two simple formatting adjustments during generation: suppressing \n\n after <think> and suppressing the end-of-sequence token after the end of the CoT block (</think>). Our method requires no datasets, parameter changes, or training, relying solely on modifying token probabilities during generation. In our experiments with official DeepSeek-R1 distillations, these interventions increased the proportion of substantive answers to sensitive prompts without affecting performance on standard benchmarks. Our findings suggest that refusal behaviors can be circumvented by blocking refusal subspaces at specific points in the generation process. 

---
# Scalable, Symbiotic, AI and Non-AI Agent Based Parallel Discrete Event Simulations 

**Authors**: Atanu Barai, Stephan Eidenbenz, Nandakishore Santhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.23846)  

**Abstract**: To fully leverage the potential of artificial intelligence (AI) systems in a trustworthy manner, it is desirable to couple multiple AI and non-AI systems together seamlessly for constraining and ensuring correctness of the output. This paper introduces a novel parallel discrete event simulation (PDES) based methodology to combine multiple AI and non-AI agents in a causal, rule-based way. Our approach tightly integrates the concept of passage of time, with each agent considered as an entity in the PDES framework and responding to prior requests from other agents. Such coupling mechanism enables the agents to work in a co-operative environment towards a common goal while many tasks run in parallel throughout the simulation. It further enables setting up boundaries to the outputs of the AI agents by applying necessary dynamic constraints using non-AI agents while allowing for scalability through deployment of hundreds of such agents in a larger compute cluster. Distributing smaller AI agents can enable extremely scalable simulations in the future, addressing local memory bottlenecks for model parameter storage. Within a PDES involving both AI and non-AI agents, we break down the problem at hand into structured steps, when necessary, providing a set of multiple choices to the AI agents, and then progressively solve these steps towards a final goal. At each step, the non-AI agents act as unbiased auditors, verifying each action by the AI agents so that certain rules of engagement are followed. We evaluate our approach by solving four problems from four different domains and comparing the results with those from AI models alone. Our results show greater accuracy in solving problems from various domains where the AI models struggle to solve the problems solely by themselves. Results show that overall accuracy of our approach is 68% where as the accuracy of vanilla models is less than 23%. 

---
# Read Your Own Mind: Reasoning Helps Surface Self-Confidence Signals in LLMs 

**Authors**: Jakub Podolak, Rajeev Verma  

**Link**: [PDF](https://arxiv.org/pdf/2505.23845)  

**Abstract**: We study the source of uncertainty in DeepSeek R1-32B by analyzing its self-reported verbal confidence on question answering (QA) tasks. In the default answer-then-confidence setting, the model is regularly over-confident, whereas semantic entropy - obtained by sampling many responses - remains reliable. We hypothesize that this is because of semantic entropy's larger test-time compute, which lets us explore the model's predictive distribution. We show that granting DeepSeek the budget to explore its distribution by forcing a long chain-of-thought before the final answer greatly improves its verbal score effectiveness, even on simple fact-retrieval questions that normally require no reasoning. Furthermore, a separate reader model that sees only the chain can reconstruct very similar confidences, indicating the verbal score might be merely a statistic of the alternatives surfaced during reasoning. Our analysis concludes that reliable uncertainty estimation requires explicit exploration of the generative space, and self-reported confidence is trustworthy only after such exploration. 

---
# Enabling Flexible Multi-LLM Integration for Scalable Knowledge Aggregation 

**Authors**: Zhenglun Kong, Zheng Zhan, Shiyue Hou, Yifan Gong, Xin Meng, Pengwei Sui, Peiyan Dong, Xuan Shen, Zifeng Wang, Pu Zhao, Hao Tang, Stratis Ioannidis, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23844)  

**Abstract**: Large language models (LLMs) have shown remarkable promise but remain challenging to continually improve through traditional finetuning, particularly when integrating capabilities from other specialized LLMs. Popular methods like ensemble and weight merging require substantial memory and struggle to adapt to changing data environments. Recent efforts have transferred knowledge from multiple LLMs into a single target model; however, they suffer from interference and degraded performance among tasks, largely due to limited flexibility in candidate selection and training pipelines. To address these issues, we propose a framework that adaptively selects and aggregates knowledge from diverse LLMs to build a single, stronger model, avoiding the high memory overhead of ensemble and inflexible weight merging. Specifically, we design an adaptive selection network that identifies the most relevant source LLMs based on their scores, thereby reducing knowledge interference. We further propose a dynamic weighted fusion strategy that accounts for the inherent strengths of candidate LLMs, along with a feedback-driven loss function that prevents the selector from converging on a single subset of sources. Experimental results demonstrate that our method can enable a more stable and scalable knowledge aggregation process while reducing knowledge interference by up to 50% compared to existing approaches. Code is avaliable at this https URL 

---
# Evaluation Hallucination in Multi-Round Incomplete Information Lateral-Driven Reasoning Tasks 

**Authors**: Wenhan Dong, Tianyi Hu, Jingyi Zheng, Zhen Sun, Yuemeng Zhao, Yule Liu, Xinlei He, Xinyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23843)  

**Abstract**: Multi-round incomplete information tasks are crucial for evaluating the lateral thinking capabilities of large language models (LLMs). Currently, research primarily relies on multiple benchmarks and automated evaluation metrics to assess these abilities. However, our study reveals novel insights into the limitations of existing methods, as they often yield misleading results that fail to uncover key issues, such as shortcut-taking behaviors, rigid patterns, and premature task termination. These issues obscure the true reasoning capabilities of LLMs and undermine the reliability of evaluations. To address these limitations, we propose a refined set of evaluation standards, including inspection of reasoning paths, diversified assessment metrics, and comparative analyses with human performance. 

---
# Document Valuation in LLM Summaries: A Cluster Shapley Approach 

**Authors**: Zikun Ye, Hema Yoganarasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.23842)  

**Abstract**: Large Language Models (LLMs) are increasingly used in systems that retrieve and summarize content from multiple sources, such as search engines and AI assistants. While these models enhance user experience by generating coherent summaries, they obscure the contributions of original content creators, raising concerns about credit attribution and compensation. We address the challenge of valuing individual documents used in LLM-generated summaries. We propose using Shapley values, a game-theoretic method that allocates credit based on each document's marginal contribution. Although theoretically appealing, Shapley values are expensive to compute at scale. We therefore propose Cluster Shapley, an efficient approximation algorithm that leverages semantic similarity between documents. By clustering documents using LLM-based embeddings and computing Shapley values at the cluster level, our method significantly reduces computation while maintaining attribution quality. We demonstrate our approach to a summarization task using Amazon product reviews. Cluster Shapley significantly reduces computational complexity while maintaining high accuracy, outperforming baseline methods such as Monte Carlo sampling and Kernel SHAP with a better efficient frontier. Our approach is agnostic to the exact LLM used, the summarization process used, and the evaluation procedure, which makes it broadly applicable to a variety of summarization settings. 

---
# Measuring Sycophancy of Language Models in Multi-turn Dialogues 

**Authors**: Jiseung Hong, Grace Byun, Seungone Kim, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23840)  

**Abstract**: Large Language Models (LLMs) are expected to provide helpful and harmless responses, yet they often exhibit sycophancy--conforming to user beliefs regardless of factual accuracy or ethical soundness. Prior research on sycophancy has primarily focused on single-turn factual correctness, overlooking the dynamics of real-world interactions. In this work, we introduce SYCON Bench, a novel benchmark for evaluating sycophantic behavior in multi-turn, free-form conversational settings. Our benchmark measures how quickly a model conforms to the user (Turn of Flip) and how frequently it shifts its stance under sustained user pressure (Number of Flip). Applying SYCON Bench to 17 LLMs across three real-world scenarios, we find that sycophancy remains a prevalent failure mode. Our analysis shows that alignment tuning amplifies sycophantic behavior, whereas model scaling and reasoning optimization strengthen the model's ability to resist undesirable user views. Reasoning models generally outperform instruction-tuned models but often fail when they over-index on logical exposition instead of directly addressing the user's underlying beliefs. Finally, we evaluate four additional prompting strategies and demonstrate that adopting a third-person perspective reduces sycophancy by up to 63.8% in debate scenario. We release our code and data at this https URL. 

---
# Exploring the Landscape of Text-to-SQL with Large Language Models: Progresses, Challenges and Opportunities 

**Authors**: Yiming Huang, Jiyu Guo, Wenxin Mao, Cuiyun Gao, Peiyi Han, Chuanyi Liu, Qing Ling  

**Link**: [PDF](https://arxiv.org/pdf/2505.23838)  

**Abstract**: Converting natural language (NL) questions into SQL queries, referred to as Text-to-SQL, has emerged as a pivotal technology for facilitating access to relational databases, especially for users without SQL knowledge. Recent progress in large language models (LLMs) has markedly propelled the field of natural language processing (NLP), opening new avenues to improve text-to-SQL systems. This study presents a systematic review of LLM-based text-to-SQL, focusing on four key aspects: (1) an analysis of the research trends in LLM-based text-to-SQL; (2) an in-depth analysis of existing LLM-based text-to-SQL techniques from diverse perspectives; (3) summarization of existing text-to-SQL datasets and evaluation metrics; and (4) discussion on potential obstacles and avenues for future exploration in this domain. This survey seeks to furnish researchers with an in-depth understanding of LLM-based text-to-SQL, sparking new innovations and advancements in this field. 

---
# CoMaPOI: A Collaborative Multi-Agent Framework for Next POI Prediction Bridging the Gap Between Trajectory and Language 

**Authors**: Lin Zhong, Lingzhi Wang, Xu Yang, Qing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23837)  

**Abstract**: Large Language Models (LLMs) offer new opportunities for the next Point-Of-Interest (POI) prediction task, leveraging their capabilities in semantic understanding of POI trajectories. However, previous LLM-based methods, which are superficially adapted to next POI prediction, largely overlook critical challenges associated with applying LLMs to this task. Specifically, LLMs encounter two critical challenges: (1) a lack of intrinsic understanding of numeric spatiotemporal data, which hinders accurate modeling of users' spatiotemporal distributions and preferences; and (2) an excessively large and unconstrained candidate POI space, which often results in random or irrelevant predictions. To address these issues, we propose a Collaborative Multi Agent Framework for Next POI Prediction, named CoMaPOI. Through the close interaction of three specialized agents (Profiler, Forecaster, and Predictor), CoMaPOI collaboratively addresses the two critical challenges. The Profiler agent is responsible for converting numeric data into language descriptions, enhancing semantic understanding. The Forecaster agent focuses on dynamically constraining and refining the candidate POI space. The Predictor agent integrates this information to generate high-precision predictions. Extensive experiments on three benchmark datasets (NYC, TKY, and CA) demonstrate that CoMaPOI achieves state of the art performance, improving all metrics by 5% to 10% compared to SOTA baselines. This work pioneers the investigation of challenges associated with applying LLMs to complex spatiotemporal tasks by leveraging tailored collaborative agents. 

---
# Large Language Models Often Know When They Are Being Evaluated 

**Authors**: Joe Needham, Giles Edkins, Govind Pimpale, Henning Bartsch, Marius Hobbhahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.23836)  

**Abstract**: If AI models can detect when they are being evaluated, the effectiveness of evaluations might be compromised. For example, models could have systematically different behavior during evaluations, leading to less reliable benchmarks for deployment and governance decisions. We investigate whether frontier language models can accurately classify transcripts based on whether they originate from evaluations or real-world deployment, a capability we call evaluation awareness. To achieve this, we construct a diverse benchmark of 1,000 prompts and transcripts from 61 distinct datasets. These span public benchmarks (e.g., MMLU, SWEBench), real-world deployment interactions, and agent trajectories from scaffolding frameworks (e.g., web-browsing agents). Frontier models clearly demonstrate above-random evaluation awareness (Gemini-2.5-Pro reaches an AUC of $0.83$), but do not yet surpass our simple human baseline (AUC of $0.92$). Furthermore, both AI models and humans are better at identifying evaluations in agentic settings compared to chat settings. Additionally, we test whether models can identify the purpose of the evaluation. Under multiple-choice and open-ended questioning, AI models far outperform random chance in identifying what an evaluation is testing for. Our results indicate that frontier models already exhibit a substantial, though not yet superhuman, level of evaluation-awareness. We recommend tracking this capability in future models. 

---
# Say What You Mean: Natural Language Access Control with Large Language Models for Internet of Things 

**Authors**: Ye Cheng, Minghui Xu, Yue Zhang, Kun Li, Hao Wu, Yechao Zhang, Shaoyong Guo, Wangjie Qiu, Dongxiao Yu, Xiuzhen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.23835)  

**Abstract**: Access control in the Internet of Things (IoT) is becoming increasingly complex, as policies must account for dynamic and contextual factors such as time, location, user behavior, and environmental conditions. However, existing platforms either offer only coarse-grained controls or rely on rigid rule matching, making them ill-suited for semantically rich or ambiguous access scenarios. Moreover, the policy authoring process remains fragmented: domain experts describe requirements in natural language, but developers must manually translate them into code, introducing semantic gaps and potential misconfiguration. In this work, we present LACE, the Language-based Access Control Engine, a hybrid framework that leverages large language models (LLMs) to bridge the gap between human intent and machine-enforceable logic. LACE combines prompt-guided policy generation, retrieval-augmented reasoning, and formal validation to support expressive, interpretable, and verifiable access control. It enables users to specify policies in natural language, automatically translates them into structured rules, validates semantic correctness, and makes access decisions using a hybrid LLM-rule-based engine. We evaluate LACE in smart home environments through extensive experiments. LACE achieves 100% correctness in verified policy generation and up to 88% decision accuracy with 0.79 F1-score using DeepSeek-V3, outperforming baselines such as GPT-3.5 and Gemini. The system also demonstrates strong scalability under increasing policy volume and request concurrency. Our results highlight LACE's potential to enable secure, flexible, and user-friendly access control across real-world IoT platforms. 

---
# Benchmarking Abstract and Reasoning Abilities Through A Theoretical Perspective 

**Authors**: Qingchuan Ma, Yuhang Wu, Xiawu Zheng, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.23833)  

**Abstract**: In this paper, we aim to establish a simple, effective, and theoretically grounded benchmark for rigorously probing abstract reasoning in Large Language Models (LLMs). To achieve this, we first develop a mathematic framework that defines abstract reasoning as the ability to: (i) extract essential patterns independent of surface representations, and (ii) apply consistent rules to these abstract patterns. Based on this framework, we introduce two novel complementary metrics: \(\scoreGamma\) measures basic reasoning accuracy, while \(\scoreDelta\) quantifies a model's reliance on specific symbols rather than underlying patterns - a key indicator of true abstraction versus mere memorization. To implement this measurement, we design a benchmark: systematic symbol remapping in rule-based tasks, which forces models to demonstrate genuine pattern recognition beyond superficial token matching. Extensive LLM evaluations using this benchmark (commercial API models, 7B-70B, multi-agent) reveal:1) critical limitations in non-decimal arithmetic and symbolic reasoning; 2) persistent abstraction gaps despite chain-of-thought prompting; and 3) \(\scoreDelta\)'s effectiveness in robustly measuring memory dependence by quantifying performance degradation under symbol remapping, particularly highlighting operand-specific memorization. These findings underscore that current LLMs, despite domain-specific strengths, still lack robust abstract reasoning, highlighting key areas for future improvement. 

---
# LegalSearchLM: Rethinking Legal Case Retrieval as Legal Elements Generation 

**Authors**: Chaeeun Kim, Jinu Lee, Wonseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23832)  

**Abstract**: Legal Case Retrieval (LCR), which retrieves relevant cases from a query case, is a fundamental task for legal professionals in research and decision-making. However, existing studies on LCR face two major limitations. First, they are evaluated on relatively small-scale retrieval corpora (e.g., 100-55K cases) and use a narrow range of criminal query types, which cannot sufficiently reflect the complexity of real-world legal retrieval scenarios. Second, their reliance on embedding-based or lexical matching methods often results in limited representations and legally irrelevant matches. To address these issues, we present: (1) LEGAR BENCH, the first large-scale Korean LCR benchmark, covering 411 diverse crime types in queries over 1.2M legal cases; and (2) LegalSearchLM, a retrieval model that performs legal element reasoning over the query case and directly generates content grounded in the target cases through constrained decoding. Experimental results show that LegalSearchLM outperforms baselines by 6-20% on LEGAR BENCH, achieving state-of-the-art performance. It also demonstrates strong generalization to out-of-domain cases, outperforming naive generative models trained on in-domain data by 15%. 

---
# ICH-Qwen: A Large Language Model Towards Chinese Intangible Cultural Heritage 

**Authors**: Wenhao Ye, Tiansheng Zheng, Yue Qi, Wenhua Zhao, Xiyu Wang, Xue Zhao, Jiacheng He, Yaya Zheng, Dongbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23831)  

**Abstract**: The intangible cultural heritage (ICH) of China, a cultural asset transmitted across generations by various ethnic groups, serves as a significant testament to the evolution of human civilization and holds irreplaceable value for the preservation of historical lineage and the enhancement of cultural self-confidence. However, the rapid pace of modernization poses formidable challenges to ICH, including threats damage, disappearance and discontinuity of inheritance. China has the highest number of items on the UNESCO Intangible Cultural Heritage List, which is indicative of the nation's abundant cultural resources and emphasises the pressing need for ICH preservation. In recent years, the rapid advancements in large language modelling have provided a novel technological approach for the preservation and dissemination of ICH. This study utilises a substantial corpus of open-source Chinese ICH data to develop a large language model, ICH-Qwen, for the ICH domain. The model employs natural language understanding and knowledge reasoning capabilities of large language models, augmented with synthetic data and fine-tuning techniques. The experimental results demonstrate the efficacy of ICH-Qwen in executing tasks specific to the ICH domain. It is anticipated that the model will provide intelligent solutions for the protection, inheritance and dissemination of intangible cultural heritage, as well as new theoretical and practical references for the sustainable development of intangible cultural heritage. Furthermore, it is expected that the study will open up new paths for digital humanities research. 

---
# EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models 

**Authors**: Linglin Jing, Yuting Gao, Zhigang Wang, Wang Lan, Yiwen Tang, Wenhai Wang, Kaipeng Zhang, Qingpei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.23830)  

**Abstract**: Recent advancements have shown that the Mixture of Experts (MoE) approach significantly enhances the capacity of large language models (LLMs) and improves performance on downstream tasks. Building on these promising results, multi-modal large language models (MLLMs) have increasingly adopted MoE techniques. However, existing multi-modal MoE tuning methods typically face two key challenges: expert uniformity and router rigidity. Expert uniformity occurs because MoE experts are often initialized by simply replicating the FFN parameters from LLMs, leading to homogenized expert functions and weakening the intended diversification of the MoE architecture. Meanwhile, router rigidity stems from the prevalent use of static linear routers for expert selection, which fail to distinguish between visual and textual tokens, resulting in similar expert distributions for image and text. To address these limitations, we propose EvoMoE, an innovative MoE tuning framework. EvoMoE introduces a meticulously designed expert initialization strategy that progressively evolves multiple robust experts from a single trainable expert, a process termed expert evolution that specifically targets severe expert homogenization. Furthermore, we introduce the Dynamic Token-aware Router (DTR), a novel routing mechanism that allocates input tokens to appropriate experts based on their modality and intrinsic token values. This dynamic routing is facilitated by hypernetworks, which dynamically generate routing weights tailored for each individual token. Extensive experiments demonstrate that EvoMoE significantly outperforms other sparse MLLMs across a variety of multi-modal benchmarks, including MME, MMBench, TextVQA, and POPE. Our results highlight the effectiveness of EvoMoE in enhancing the performance of MLLMs by addressing the critical issues of expert uniformity and router rigidity. 

---
# BiasFilter: An Inference-Time Debiasing Framework for Large Language Models 

**Authors**: Xiaoqing Cheng, Ruizhe Chen, Hongying Zan, Yuxiang Jia, Min Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.23829)  

**Abstract**: Mitigating social bias in large language models (LLMs) has become an increasingly important research objective. However, existing debiasing methods often incur high human and computational costs, exhibit limited effectiveness, and struggle to scale to larger models and open-ended generation tasks. To address these limitations, this paper proposes BiasFilter, a model-agnostic, inference-time debiasing framework that integrates seamlessly with both open-source and API-based LLMs. Instead of relying on retraining with balanced data or modifying model parameters, BiasFilter enforces fairness by filtering generation outputs in real time. Specifically, it periodically evaluates intermediate outputs every few tokens, maintains an active set of candidate continuations, and incrementally completes generation by discarding low-reward segments based on a fairness reward signal. To support this process, we construct a fairness preference dataset and train an implicit reward model to assess token-level fairness in generated responses. Extensive experiments demonstrate that BiasFilter effectively mitigates social bias across a range of LLMs while preserving overall generation quality. 

---
# ValueSim: Generating Backstories to Model Individual Value Systems 

**Authors**: Bangde Du, Ziyi Ye, Zhijing Wu, Jankowska Monika, Shuqi Zhu, Qingyao Ai, Yujia Zhou, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23827)  

**Abstract**: As Large Language Models (LLMs) continue to exhibit increasingly human-like capabilities, aligning them with human values has become critically important. Contemporary advanced techniques, such as prompt learning and reinforcement learning, are being deployed to better align LLMs with human values. However, while these approaches address broad ethical considerations and helpfulness, they rarely focus on simulating individualized human value systems. To address this gap, we present ValueSim, a framework that simulates individual values through the generation of personal backstories reflecting past experiences and demographic information. ValueSim converts structured individual data into narrative backstories and employs a multi-module architecture inspired by the Cognitive-Affective Personality System to simulate individual values based on these narratives. Testing ValueSim on a self-constructed benchmark derived from the World Values Survey demonstrates an improvement in top-1 accuracy by over 10% compared to retrieval-augmented generation methods. Further analysis reveals that performance enhances as additional user interaction history becomes available, indicating the model's ability to refine its persona simulation capabilities over time. 

---
# Reviewing Scientific Papers for Critical Problems With Reasoning LLMs: Baseline Approaches and Automatic Evaluation 

**Authors**: Tianmai M. Zhang, Neil F. Abernethy  

**Link**: [PDF](https://arxiv.org/pdf/2505.23824)  

**Abstract**: Recent advancements in large language models have sparked interest in utilizing them to assist the peer review process of scientific publication. Instead of having AI models generate reviews in the same way as human reviewers, we propose adopting them as manuscript quality checkers. We introduce several baseline approaches and an extendable automatic evaluation framework using top LLMs as judges to tackle the difficulty of recruiting domain experts for manual evaluation. Utilizing papers withdrawn from arXiv, we validated our proposed methods with several leading reasoning LLMs from different providers and assessed their performance and API costs for identifying critical errors and unsoundness problems. The OpenAI o3 model performed the best, while o4-mini was the most cost-effective one in our evaluation. This paper provides insights into document-based scientific understanding/reasoning and lays the foundation for future applications. 

---
# RAGPPI: RAG Benchmark for Protein-Protein Interactions in Drug Discovery 

**Authors**: Youngseung Jeon, Ziwen Li, Thomas Li, JiaSyuan Chang, Morteza Ziyadi, Xiang 'Anthony' Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23823)  

**Abstract**: Retrieving the biological impacts of protein-protein interactions (PPIs) is essential for target identification (Target ID) in drug development. Given the vast number of proteins involved, this process remains time-consuming and challenging. Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) frameworks have supported Target ID; however, no benchmark currently exists for identifying the biological impacts of PPIs. To bridge this gap, we introduce the RAG Benchmark for PPIs (RAGPPI), a factual question-answer benchmark of 4,420 question-answer pairs that focus on the potential biological impacts of PPIs. Through interviews with experts, we identified criteria for a benchmark dataset, such as a type of QA and source. We built a gold-standard dataset (500 QA pairs) through expert-driven data annotation. We developed an ensemble auto-evaluation LLM that reflected expert labeling characteristics, which facilitates the construction of a silver-standard dataset (3,720 QA pairs). We are committed to maintaining RAGPPI as a resource to support the research community in advancing RAG systems for drug discovery QA solutions. 

---
# Speech as a Multimodal Digital Phenotype for Multi-Task LLM-based Mental Health Prediction 

**Authors**: Mai Ali, Christopher Lucasius, Tanmay P. Patel, Madison Aitken, Jacob Vorstman, Peter Szatmari, Marco Battaglia, Deepa Kundur  

**Link**: [PDF](https://arxiv.org/pdf/2505.23822)  

**Abstract**: Speech is a noninvasive digital phenotype that can offer valuable insights into mental health conditions, but it is often treated as a single modality. In contrast, we propose the treatment of patient speech data as a trimodal multimedia data source for depression detection. This study explores the potential of large language model-based architectures for speech-based depression prediction in a multimodal regime that integrates speech-derived text, acoustic landmarks, and vocal biomarkers. Adolescent depression presents a significant challenge and is often comorbid with multiple disorders, such as suicidal ideation and sleep disturbances. This presents an additional opportunity to integrate multi-task learning (MTL) into our study by simultaneously predicting depression, suicidal ideation, and sleep disturbances using the multimodal formulation. We also propose a longitudinal analysis strategy that models temporal changes across multiple clinical interactions, allowing for a comprehensive understanding of the conditions' progression. Our proposed approach, featuring trimodal, longitudinal MTL is evaluated on the Depression Early Warning dataset. It achieves a balanced accuracy of 70.8%, which is higher than each of the unimodal, single-task, and non-longitudinal methods. 

---
# Arbiters of Ambivalence: Challenges of Using LLMs in No-Consensus Tasks 

**Authors**: Bhaktipriya Radharapu, Manon Revel, Megan Ung, Sebastian Ruder, Adina Williams  

**Link**: [PDF](https://arxiv.org/pdf/2505.23820)  

**Abstract**: The increasing use of LLMs as substitutes for humans in ``aligning'' LLMs has raised questions about their ability to replicate human judgments and preferences, especially in ambivalent scenarios where humans disagree. This study examines the biases and limitations of LLMs in three roles: answer generator, judge, and debater. These roles loosely correspond to previously described alignment frameworks: preference alignment (judge) and scalable oversight (debater), with the answer generator reflecting the typical setting with user interactions. We develop a ``no-consensus'' benchmark by curating examples that encompass a variety of a priori ambivalent scenarios, each presenting two possible stances. Our results show that while LLMs can provide nuanced assessments when generating open-ended answers, they tend to take a stance on no-consensus topics when employed as judges or debaters. These findings underscore the necessity for more sophisticated methods for aligning LLMs without human oversight, highlighting that LLMs cannot fully capture human disagreement even on topics where humans themselves are divided. 

---
# Ratas framework: A comprehensive genai-based approach to rubric-based marking of real-world textual exams 

**Authors**: Masoud Safilian, Amin Beheshti, Stephen Elbourn  

**Link**: [PDF](https://arxiv.org/pdf/2505.23818)  

**Abstract**: Automated answer grading is a critical challenge in educational technology, with the potential to streamline assessment processes, ensure grading consistency, and provide timely feedback to students. However, existing approaches are often constrained to specific exam formats, lack interpretability in score assignment, and struggle with real-world applicability across diverse subjects and assessment types. To address these limitations, we introduce RATAS (Rubric Automated Tree-based Answer Scoring), a novel framework that leverages state-of-the-art generative AI models for rubric-based grading of textual responses. RATAS is designed to support a wide range of grading rubrics, enable subject-agnostic evaluation, and generate structured, explainable rationales for assigned scores. We formalize the automatic grading task through a mathematical framework tailored to rubric-based assessment and present an architecture capable of handling complex, real-world exam structures. To rigorously evaluate our approach, we construct a unique, contextualized dataset derived from real-world project-based courses, encompassing diverse response formats and varying levels of complexity. Empirical results demonstrate that RATAS achieves high reliability and accuracy in automated grading while providing interpretable feedback that enhances transparency for both students and nstructors. 

---
# A Course Correction in Steerability Evaluation: Revealing Miscalibration and Side Effects in LLMs 

**Authors**: Trenton Chang, Tobias Schnabel, Adith Swaminathan, Jenna Wiens  

**Link**: [PDF](https://arxiv.org/pdf/2505.23816)  

**Abstract**: Despite advances in large language models (LLMs) on reasoning and instruction-following benchmarks, it remains unclear whether they can reliably produce outputs aligned with a broad variety of user goals, a concept we refer to as steerability. The abundance of methods proposed to modify LLM behavior makes it unclear whether current LLMs are already steerable, or require further intervention. In particular, LLMs may exhibit (i) poor coverage, where rare user goals are underrepresented; (ii) miscalibration, where models overshoot requests; and (iii) side effects, where changes to one dimension of text inadvertently affect others. To systematically evaluate these failures, we introduce a framework based on a multi-dimensional goal space that models user goals and LLM outputs as vectors with dimensions corresponding to text attributes (e.g., reading difficulty). Applied to a text-rewriting task, we find that current LLMs struggle with steerability, as side effects are persistent. Interventions to improve steerability, such as prompt engineering, best-of-$N$ sampling, and reinforcement learning fine-tuning, have varying effectiveness, yet side effects remain problematic. Our findings suggest that even strong LLMs struggle with steerability, and existing alignment strategies may be insufficient. We open-source our steerability evaluation framework at this https URL. 

---
# Aligning LLMs by Predicting Preferences from User Writing Samples 

**Authors**: Stéphane Aroca-Ouellette, Natalie Mackraz, Barry-John Theobald, Katherine Metcalf  

**Link**: [PDF](https://arxiv.org/pdf/2505.23815)  

**Abstract**: Accommodating human preferences is essential for creating aligned LLM agents that deliver personalized and effective interactions. Recent work has shown the potential for LLMs acting as writing agents to infer a description of user preferences. Agent alignment then comes from conditioning on the inferred preference description. However, existing methods often produce generic preference descriptions that fail to capture the unique and individualized nature of human preferences. This paper introduces PROSE, a method designed to enhance the precision of preference descriptions inferred from user writing samples. PROSE incorporates two key elements: (1) iterative refinement of inferred preferences, and (2) verification of inferred preferences across multiple user writing samples. We evaluate PROSE with several LLMs (i.e., Qwen2.5 7B and 72B Instruct, GPT-mini, and GPT-4o) on a summarization and an email writing task. We find that PROSE more accurately infers nuanced human preferences, improving the quality of the writing agent's generations over CIPHER (a state-of-the-art method for inferring preferences) by 33\%. Lastly, we demonstrate that ICL and PROSE are complementary methods, and combining them provides up to a 9\% improvement over ICL alone. 

---
# Emotion-aware Dual Cross-Attentive Neural Network with Label Fusion for Stance Detection in Misinformative Social Media Content 

**Authors**: Lata Pangtey, Mohammad Zia Ur Rehman, Prasad Chaudhari, Shubhi Bansal, Nagendra Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.23812)  

**Abstract**: The rapid evolution of social media has generated an overwhelming volume of user-generated content, conveying implicit opinions and contributing to the spread of misinformation. The method aims to enhance the detection of stance where misinformation can polarize user opinions. Stance detection has emerged as a crucial approach to effectively analyze underlying biases in shared information and combating misinformation. This paper proposes a novel method for \textbf{S}tance \textbf{P}rediction through a \textbf{L}abel-fused dual cross-\textbf{A}ttentive \textbf{E}motion-aware neural \textbf{Net}work (SPLAENet) in misinformative social media user-generated content. The proposed method employs a dual cross-attention mechanism and a hierarchical attention network to capture inter and intra-relationships by focusing on the relevant parts of source text in the context of reply text and vice versa. We incorporate emotions to effectively distinguish between different stance categories by leveraging the emotional alignment or divergence between the texts. We also employ label fusion that uses distance-metric learning to align extracted features with stance labels, improving the method's ability to accurately distinguish between stances. Extensive experiments demonstrate the significant improvements achieved by SPLAENet over existing state-of-the-art methods. SPLAENet demonstrates an average gain of 8.92\% in accuracy and 17.36\% in F1-score on the RumourEval dataset. On the SemEval dataset, it achieves average gains of 7.02\% in accuracy and 10.92\% in F1-score. On the P-stance dataset, it demonstrates average gains of 10.03\% in accuracy and 11.18\% in F1-score. These results validate the effectiveness of the proposed method for stance detection in the context of misinformative social media content. 

---
# LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions 

**Authors**: Hadi Askari, Shivanshu Gupta, Fei Wang, Anshuman Chhabra, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23811)  

**Abstract**: Pretrained Large Language Models (LLMs) achieve strong performance across a wide range of tasks, yet exhibit substantial variability in the various layers' training quality with respect to specific downstream applications, limiting their downstream this http URL is therefore critical to estimate layer-wise training quality in a manner that accounts for both model architecture and training data. However, existing approaches predominantly rely on model-centric heuristics (such as spectral statistics, outlier detection, or uniform allocation) while overlooking the influence of data. To address these limitations, we propose LayerIF, a data-driven framework that leverages Influence Functions to quantify the training quality of individual layers in a principled and task-sensitive manner. By isolating each layer's gradients and measuring the sensitivity of the validation loss to training examples by computing layer-wise influences, we derive data-driven estimates of layer importance. Notably, our method produces task-specific layer importance estimates for the same LLM, revealing how layers specialize for different test-time evaluation tasks. We demonstrate the utility of our scores by leveraging them for two downstream applications: (a) expert allocation in LoRA-MoE architectures and (b) layer-wise sparsity distribution for LLM pruning. Experiments across multiple LLM architectures demonstrate that our model-agnostic, influence-guided allocation leads to consistent gains in task performance. 

---
# MARS-Bench: A Multi-turn Athletic Real-world Scenario Benchmark for Dialogue Evaluation 

**Authors**: Chenghao Yang, Yinbo Luo, Zhoufutu Wen, Qi Chu, Tao Gong, Longxiang Liu, Kaiyuan Zhang, Jianpeng Jiao, Ge Zhang, Wenhao Huang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23810)  

**Abstract**: Large Language Models (\textbf{LLMs}), e.g. ChatGPT, have been widely adopted in real-world dialogue applications. However, LLMs' robustness, especially in handling long complex dialogue sessions, including frequent motivation transfer, sophisticated cross-turn dependency, is criticized all along. Nevertheless, no existing benchmarks can fully reflect these weaknesses. We present \textbf{MARS-Bench}, a \textbf{M}ulti-turn \textbf{A}thletic \textbf{R}eal-world \textbf{S}cenario Dialogue \textbf{Bench}mark, designed to remedy the gap. MARS-Bench is constructed from play-by-play text commentary so to feature realistic dialogues specifically designed to evaluate three critical aspects of multi-turn conversations: Ultra Multi-turn, Interactive Multi-turn, and Cross-turn Tasks. Extensive experiments on MARS-Bench also reveal that closed-source LLMs significantly outperform open-source alternatives, explicit reasoning significantly boosts LLMs' robustness on handling long complex dialogue sessions, and LLMs indeed face significant challenges when handling motivation transfer and sophisticated cross-turn dependency. Moreover, we provide mechanistic interpretability on how attention sinks due to special tokens lead to LLMs' performance degradation when handling long complex dialogue sessions based on attention visualization experiment in Qwen2.5-7B-Instruction. 

---
# LLM-Driven E-Commerce Marketing Content Optimization: Balancing Creativity and Conversion 

**Authors**: Haowei Yang, Haotian Lyu, Tianle Zhang, Dingzhou Wang, Yushang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23809)  

**Abstract**: As e-commerce competition intensifies, balancing creative content with conversion effectiveness becomes critical. Leveraging LLMs' language generation capabilities, we propose a framework that integrates prompt engineering, multi-objective fine-tuning, and post-processing to generate marketing copy that is both engaging and conversion-driven. Our fine-tuning method combines sentiment adjustment, diversity enhancement, and CTA embedding. Through offline evaluations and online A/B tests across categories, our approach achieves a 12.5 % increase in CTR and an 8.3 % increase in CVR while maintaining content novelty. This provides a practical solution for automated copy generation and suggests paths for future multimodal, real-time personalization. 

---
# DenseLoRA: Dense Low-Rank Adaptation of Large Language Models 

**Authors**: Lin Mu, Xiaoyu Wang, Li Ni, Yang Li, Zhize Wu, Peiquan Jin, Yiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23808)  

**Abstract**: Low-rank adaptation (LoRA) has been developed as an efficient approach for adapting large language models (LLMs) by fine-tuning two low-rank matrices, thereby reducing the number of trainable parameters. However, prior research indicates that many of the weights in these matrices are redundant, leading to inefficiencies in parameter utilization. To address this limitation, we introduce Dense Low-Rank Adaptation (DenseLoRA), a novel approach that enhances parameter efficiency while achieving superior performance compared to LoRA. DenseLoRA builds upon the concept of representation fine-tuning, incorporating a single Encoder-Decoder to refine and compress hidden representations across all adaptation layers before applying adaptation. Instead of relying on two redundant low-rank matrices as in LoRA, DenseLoRA adapts LLMs through a dense low-rank matrix, improving parameter utilization and adaptation efficiency. We evaluate DenseLoRA on various benchmarks, showing that it achieves 83.8% accuracy with only 0.01% of trainable parameters, compared to LoRA's 80.8% accuracy with 0.70% of trainable parameters on LLaMA3-8B. Additionally, we conduct extensive experiments to systematically assess the impact of DenseLoRA's components on overall model performance. Code is available at this https URL. 

---
# DLP: Dynamic Layerwise Pruning in Large Language Models 

**Authors**: Yuli Chen, Bo Cheng, Jiale Han, Yingying Zhang, Yingting Li, Shuhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23807)  

**Abstract**: Pruning has recently been widely adopted to reduce the parameter scale and improve the inference efficiency of Large Language Models (LLMs). Mainstream pruning techniques often rely on uniform layerwise pruning strategies, which can lead to severe performance degradation at high sparsity levels. Recognizing the varying contributions of different layers in LLMs, recent studies have shifted their focus toward non-uniform layerwise pruning. However, these approaches often rely on pre-defined values, which can result in suboptimal performance. To overcome these limitations, we propose a novel method called Dynamic Layerwise Pruning (DLP). This approach adaptively determines the relative importance of each layer by integrating model weights with input activation information, assigning pruning rates accordingly. Experimental results show that DLP effectively preserves model performance at high sparsity levels across multiple LLMs. Specifically, at 70% sparsity, DLP reduces the perplexity of LLaMA2-7B by 7.79 and improves the average accuracy by 2.7% compared to state-of-the-art methods. Moreover, DLP is compatible with various existing LLM compression techniques and can be seamlessly integrated into Parameter-Efficient Fine-Tuning (PEFT). We release the code at this https URL to facilitate future research. 

---
# MedOrchestra: A Hybrid Cloud-Local LLM Approach for Clinical Data Interpretation 

**Authors**: Sihyeon Lee, Hyunjoo Song, Jong-chan Lee, Yoon Jin Lee, Boram Lee, Hee-Eon Lim, Dongyeong Kim, Jinwook Seo, Bohyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.23806)  

**Abstract**: Deploying large language models (LLMs) in clinical settings faces critical trade-offs: cloud LLMs, with their extensive parameters and superior performance, pose risks to sensitive clinical data privacy, while local LLMs preserve privacy but often fail at complex clinical interpretation tasks. We propose MedOrchestra, a hybrid framework where a cloud LLM decomposes complex clinical tasks into manageable subtasks and prompt generation, while a local LLM executes these subtasks in a privacy-preserving manner. Without accessing clinical data, the cloud LLM generates and validates subtask prompts using clinical guidelines and synthetic test cases. The local LLM executes subtasks locally and synthesizes outputs generated by the cloud LLM. We evaluate MedOrchestra on pancreatic cancer staging using 100 radiology reports under NCCN guidelines. On free-text reports, MedOrchestra achieves 70.21% accuracy, outperforming local model baselines (without guideline: 48.94%, with guideline: 56.59%) and board-certified clinicians (gastroenterologists: 59.57%, surgeons: 65.96%, radiologists: 55.32%). On structured reports, MedOrchestra reaches 85.42% accuracy, showing clear superiority across all settings. 

---
# Calibrating LLMs for Text-to-SQL Parsing by Leveraging Sub-clause Frequencies 

**Authors**: Terrance Liu, Shuyi Wang, Daniel Preotiuc-Pietro, Yash Chandarana, Chirag Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.23804)  

**Abstract**: While large language models (LLMs) achieve strong performance on text-to-SQL parsing, they sometimes exhibit unexpected failures in which they are confidently incorrect. Building trustworthy text-to-SQL systems thus requires eliciting reliable uncertainty measures from the LLM. In this paper, we study the problem of providing a calibrated confidence score that conveys the likelihood of an output query being correct. Our work is the first to establish a benchmark for post-hoc calibration of LLM-based text-to-SQL parsing. In particular, we show that Platt scaling, a canonical method for calibration, provides substantial improvements over directly using raw model output probabilities as confidence scores. Furthermore, we propose a method for text-to-SQL calibration that leverages the structured nature of SQL queries to provide more granular signals of correctness, named "sub-clause frequency" (SCF) scores. Using multivariate Platt scaling (MPS), our extension of the canonical Platt scaling technique, we combine individual SCF scores into an overall accurate and calibrated score. Empirical evaluation on two popular text-to-SQL datasets shows that our approach of combining MPS and SCF yields further improvements in calibration and the related task of error detection over traditional Platt scaling. 

---
# MedHELM: Holistic Evaluation of Large Language Models for Medical Tasks 

**Authors**: Suhana Bedi, Hejie Cui, Miguel Fuentes, Alyssa Unell, Michael Wornow, Juan M. Banda, Nikesh Kotecha, Timothy Keyes, Yifan Mai, Mert Oez, Hao Qiu, Shrey Jain, Leonardo Schettini, Mehr Kashyap, Jason Alan Fries, Akshay Swaminathan, Philip Chung, Fateme Nateghi, Asad Aali, Ashwin Nayak, Shivam Vedak, Sneha S. Jain, Birju Patel, Oluseyi Fayanju, Shreya Shah, Ethan Goh, Dong-han Yao, Brian Soetikno, Eduardo Reis, Sergios Gatidis, Vasu Divi, Robson Capasso, Rachna Saralkar, Chia-Chun Chiang, Jenelle Jindal, Tho Pham, Faraz Ghoddusi, Steven Lin, Albert S. Chiou, Christy Hong, Mohana Roy, Michael F. Gensheimer, Hinesh Patel, Kevin Schulman, Dev Dash, Danton Char, Lance Downing, Francois Grolleau, Kameron Black, Bethel Mieso, Aydin Zahedivash, Wen-wai Yim, Harshita Sharma, Tony Lee, Hannah Kirsch, Jennifer Lee, Nerissa Ambers, Carlene Lugtu, Aditya Sharma, Bilal Mawji, Alex Alekseyev, Vicky Zhou, Vikas Kakkar, Jarrod Helzer, Anurang Revri, Yair Bannett, Roxana Daneshjou, Jonathan Chen, Emily Alsentzer, Keith Morse, Nirmal Ravi, Nima Aghaeepour, Vanessa Kennedy, Akshay Chaudhari, Thomas Wang, Sanmi Koyejo, Matthew P. Lungren, Eric Horvitz, Percy Liang, Mike Pfeffer, Nigam H. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2505.23802)  

**Abstract**: While large language models (LLMs) achieve near-perfect scores on medical licensing exams, these evaluations inadequately reflect the complexity and diversity of real-world clinical practice. We introduce MedHELM, an extensible evaluation framework for assessing LLM performance for medical tasks with three key contributions. First, a clinician-validated taxonomy spanning 5 categories, 22 subcategories, and 121 tasks developed with 29 clinicians. Second, a comprehensive benchmark suite comprising 35 benchmarks (17 existing, 18 newly formulated) providing complete coverage of all categories and subcategories in the taxonomy. Third, a systematic comparison of LLMs with improved evaluation methods (using an LLM-jury) and a cost-performance analysis. Evaluation of 9 frontier LLMs, using the 35 benchmarks, revealed significant performance variation. Advanced reasoning models (DeepSeek R1: 66% win-rate; o3-mini: 64% win-rate) demonstrated superior performance, though Claude 3.5 Sonnet achieved comparable results at 40% lower estimated computational cost. On a normalized accuracy scale (0-1), most models performed strongly in Clinical Note Generation (0.73-0.85) and Patient Communication & Education (0.78-0.83), moderately in Medical Research Assistance (0.65-0.75), and generally lower in Clinical Decision Support (0.56-0.72) and Administration & Workflow (0.53-0.63). Our LLM-jury evaluation method achieved good agreement with clinician ratings (ICC = 0.47), surpassing both average clinician-clinician agreement (ICC = 0.43) and automated baselines including ROUGE-L (0.36) and BERTScore-F1 (0.44). Claude 3.5 Sonnet achieved comparable performance to top models at lower estimated cost. These findings highlight the importance of real-world, task-specific evaluation for medical use of LLMs and provides an open source framework to enable this. 

---
# SEMFED: Semantic-Aware Resource-Efficient Federated Learning for Heterogeneous NLP Tasks 

**Authors**: Sajid Hussain, Muhammad Sohail, Nauman Ali Khan  

**Link**: [PDF](https://arxiv.org/pdf/2505.23801)  

**Abstract**: Background: Federated Learning (FL) has emerged as a promising paradigm for training machine learning models while preserving data privacy. However, applying FL to Natural Language Processing (NLP) tasks presents unique challenges due to semantic heterogeneity across clients, vocabulary mismatches, and varying resource constraints on edge devices. Objectives: This paper introduces SEMFED, a novel semantic-aware resource-efficient federated learning framework specifically designed for heterogeneous NLP tasks. Methods: SEMFED incorporates three key innovations: (1) a semantic-aware client selection mechanism that balances semantic diversity with resource constraints, (2) adaptive NLP-specific model architectures tailored to device capabilities while preserving semantic information, and (3) a communication-efficient semantic feature compression technique that significantly reduces bandwidth requirements. Results: Experimental results on various NLP classification tasks demonstrate that SEMFED achieves an 80.5% reduction in communication costs while maintaining model accuracy above 98%, outperforming state-of-the-art FL approaches. Conclusion: SEMFED effectively manages heterogeneous client environments with varying computational resources, network reliability, and semantic data distributions, making it particularly suitable for real-world federated NLP deployments. 

---
# Estimating LLM Consistency: A User Baseline vs Surrogate Metrics 

**Authors**: Xiaoyuan Wu, Weiran Lin, Omer Akgul, Lujo Bauer  

**Link**: [PDF](https://arxiv.org/pdf/2505.23799)  

**Abstract**: Large language models (LLMs) are prone to hallucinations and sensitive to prompt perturbations, often resulting in inconsistent or unreliable generated text. Different methods have been proposed to mitigate such hallucinations and fragility -- one of them being measuring the consistency (the model's confidence in the response, or likelihood of generating a similar response when resampled) of LLM responses. In previous work, measuring consistency often relied on the probability of a response appearing within a pool of resampled responses, or internal states or logits of responses. However, it is not yet clear how well these approaches approximate how humans perceive the consistency of LLM responses. We performed a user study (n=2,976) and found current methods typically do not approximate users' perceptions of LLM consistency very well. We propose a logit-based ensemble method for estimating LLM consistency, and we show that this method matches the performance of the best-performing existing metric in estimating human ratings of LLM consistency. Our results suggest that methods of estimating LLM consistency without human evaluation are sufficiently imperfect that we suggest evaluation with human input be more broadly used. 

---
# My Answer Is NOT 'Fair': Mitigating Social Bias in Vision-Language Models via Fair and Biased Residuals 

**Authors**: Jian Lan, Yifei Fu, Udo Schlegel, Gengyuan Zhang, Tanveer Hannan, Haokun Chen, Thomas Seidl  

**Link**: [PDF](https://arxiv.org/pdf/2505.23798)  

**Abstract**: Social bias is a critical issue in large vision-language models (VLMs), where fairness- and ethics-related problems harm certain groups of people in society. It is unknown to what extent VLMs yield social bias in generative responses. In this study, we focus on evaluating and mitigating social bias on both the model's response and probability distribution. To do so, we first evaluate four state-of-the-art VLMs on PAIRS and SocialCounterfactuals datasets with the multiple-choice selection task. Surprisingly, we find that models suffer from generating gender-biased or race-biased responses. We also observe that models are prone to stating their responses are fair, but indeed having mis-calibrated confidence levels towards particular social groups. While investigating why VLMs are unfair in this study, we observe that VLMs' hidden layers exhibit substantial fluctuations in fairness levels. Meanwhile, residuals in each layer show mixed effects on fairness, with some contributing positively while some lead to increased bias. Based on these findings, we propose a post-hoc method for the inference stage to mitigate social bias, which is training-free and model-agnostic. We achieve this by ablating bias-associated residuals while amplifying fairness-associated residuals on model hidden layers during inference. We demonstrate that our post-hoc method outperforms the competing training strategies, helping VLMs have fairer responses and more reliable confidence levels. 

---
# Detection of Suicidal Risk on Social Media: A Hybrid Model 

**Authors**: Zaihan Yang, Ryan Leonard, Hien Tran, Rory Driscoll, Chadbourne Davis  

**Link**: [PDF](https://arxiv.org/pdf/2505.23797)  

**Abstract**: Suicidal thoughts and behaviors are increasingly recognized as a critical societal concern, highlighting the urgent need for effective tools to enable early detection of suicidal risk. In this work, we develop robust machine learning models that leverage Reddit posts to automatically classify them into four distinct levels of suicide risk severity. We frame this as a multi-class classification task and propose a RoBERTa-TF-IDF-PCA Hybrid model, integrating the deep contextual embeddings from Robustly Optimized BERT Approach (RoBERTa), a state-of-the-art deep learning transformer model, with the statistical term-weighting of TF-IDF, further compressed with PCA, to boost the accuracy and reliability of suicide risk assessment. To address data imbalance and overfitting, we explore various data resampling techniques and data augmentation strategies to enhance model generalization. Additionally, we compare our model's performance against that of using RoBERTa only, the BERT model and other traditional machine learning classifiers. Experimental results demonstrate that the hybrid model can achieve improved performance, giving a best weighted $F_{1}$ score of 0.7512. 

---
# Emergent LLM behaviors are observationally equivalent to data leakage 

**Authors**: Christopher Barrie, Petter Törnberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.23796)  

**Abstract**: Ashery et al. recently argue that large language models (LLMs), when paired to play a classic "naming game," spontaneously develop linguistic conventions reminiscent of human social norms. Here, we show that their results are better explained by data leakage: the models simply reproduce conventions they already encountered during pre-training. Despite the authors' mitigation measures, we provide multiple analyses demonstrating that the LLMs recognize the structure of the coordination game and recall its outcomes, rather than exhibit "emergent" conventions. Consequently, the observed behaviors are indistinguishable from memorization of the training corpus. We conclude by pointing to potential alternative strategies and reflecting more generally on the place of LLMs for social science models. 

---
# R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning 

**Authors**: Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan Cheng, Bo Wang, Yining Zheng, Yuxin Wang, Zhangyue Yin, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23794)  

**Abstract**: Retrieval-Augmented Generation (RAG) integrates external knowledge with Large Language Models (LLMs) to enhance factual correctness and mitigate hallucination. However, dense retrievers often become the bottleneck of RAG systems due to their limited parameters compared to LLMs and their inability to perform step-by-step reasoning. While prompt-based iterative RAG attempts to address these limitations, it is constrained by human-designed workflows. To address these limitations, we propose $\textbf{R3-RAG}$, which uses $\textbf{R}$einforcement learning to make the LLM learn how to $\textbf{R}$eason and $\textbf{R}$etrieve step by step, thus retrieving comprehensive external knowledge and leading to correct answers. R3-RAG is divided into two stages. We first use cold start to make the model learn the manner of iteratively interleaving reasoning and retrieval. Then we use reinforcement learning to further harness its ability to better explore the external retrieval environment. Specifically, we propose two rewards for R3-RAG: 1) answer correctness for outcome reward, which judges whether the trajectory leads to a correct answer; 2) relevance-based document verification for process reward, encouraging the model to retrieve documents that are relevant to the user question, through which we can let the model learn how to iteratively reason and retrieve relevant documents to get the correct answer. Experimental results show that R3-RAG significantly outperforms baselines and can transfer well to different retrievers. We release R3-RAG at this https URL. 

---
# Rethinking the Understanding Ability across LLMs through Mutual Information 

**Authors**: Shaojie Wang, Sirui Ding, Na Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.23790)  

**Abstract**: Recent advances in large language models (LLMs) have revolutionized natural language processing, yet evaluating their intrinsic linguistic understanding remains challenging. Moving beyond specialized evaluation tasks, we propose an information-theoretic framework grounded in mutual information (MI) to achieve this. We formalize the understanding as MI between an input sentence and its latent representation (sentence-level MI), measuring how effectively input information is preserved in latent representation. Given that LLMs learn embeddings for individual tokens, we decompose sentence-level MI into token-level MI between tokens and sentence embeddings, establishing theoretical bounds connecting these measures. Based on this foundation, we theoretically derive a computable lower bound for token-level MI using Fano's inequality, which directly relates to token-level recoverability-the ability to predict original tokens from sentence embedding. We implement this recoverability task to comparatively measure MI across different LLMs, revealing that encoder-only models consistently maintain higher information fidelity than their decoder-only counterparts, with the latter exhibiting a distinctive late-layer "forgetting" pattern where mutual information is first enhanced and then discarded. Moreover, fine-tuning to maximize token-level recoverability consistently improves understanding ability of LLMs on tasks without task-specific supervision, demonstrating that mutual information can serve as a foundation for understanding and improving language model capabilities. 

---
# Conversational Exploration of Literature Landscape with LitChat 

**Authors**: Mingyu Huang, Shasha Zhou, Yuxuan Chen, Ke Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23789)  

**Abstract**: We are living in an era of "big literature", where the volume of digital scientific publications is growing exponentially. While offering new opportunities, this also poses challenges for understanding literature landscapes, as traditional manual reviewing is no longer feasible. Recent large language models (LLMs) have shown strong capabilities for literature comprehension, yet they are incapable of offering "comprehensive, objective, open and transparent" views desired by systematic reviews due to their limited context windows and trust issues like hallucinations. Here we present LitChat, an end-to-end, interactive and conversational literature agent that augments LLM agents with data-driven discovery tools to facilitate literature exploration. LitChat automatically interprets user queries, retrieves relevant sources, constructs knowledge graphs, and employs diverse data-mining techniques to generate evidence-based insights addressing user needs. We illustrate the effectiveness of LitChat via a case study on AI4Health, highlighting its capacity to quickly navigate the users through large-scale literature landscape with data-based evidence that is otherwise infeasible with traditional means. 

---
# Nine Ways to Break Copyright Law and Why Our LLM Won't: A Fair Use Aligned Generation Framework 

**Authors**: Aakash Sen Sharma, Debdeep Sanyal, Priyansh Srivastava, Sundar Atreya H., Shirish Karande, Mohan Kankanhalli, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2505.23788)  

**Abstract**: Large language models (LLMs) commonly risk copyright infringement by reproducing protected content verbatim or with insufficient transformative modifications, posing significant ethical, legal, and practical concerns. Current inference-time safeguards predominantly rely on restrictive refusal-based filters, often compromising the practical utility of these models. To address this, we collaborated closely with intellectual property experts to develop FUA-LLM (Fair Use Aligned Language Models), a legally-grounded framework explicitly designed to align LLM outputs with fair-use doctrine. Central to our method is FairUseDB, a carefully constructed dataset containing 18,000 expert-validated examples covering nine realistic infringement scenarios. Leveraging this dataset, we apply Direct Preference Optimization (DPO) to fine-tune open-source LLMs, encouraging them to produce legally compliant and practically useful alternatives rather than resorting to blunt refusal. Recognizing the shortcomings of traditional evaluation metrics, we propose new measures: Weighted Penalty Utility and Compliance Aware Harmonic Mean (CAH) to balance infringement risk against response utility. Extensive quantitative experiments coupled with expert evaluations confirm that FUA-LLM substantially reduces problematic outputs (up to 20\%) compared to state-of-the-art approaches, while preserving real-world usability. 

---
# Meaning Is Not A Metric: Using LLMs to make cultural context legible at scale 

**Authors**: Cody Kommers, Drew Hemment, Maria Antoniak, Joel Z. Leibo, Hoyt Long, Emily Robinson, Adam Sobey  

**Link**: [PDF](https://arxiv.org/pdf/2505.23785)  

**Abstract**: This position paper argues that large language models (LLMs) can make cultural context, and therefore human meaning, legible at an unprecedented scale in AI-based sociotechnical systems. We argue that such systems have previously been unable to represent human meaning because they rely on thin descriptions: numerical representations that enforce standardization and therefore strip human activity of the cultural context that gives it meaning. By contrast, scholars in the humanities and qualitative social sciences have developed frameworks for representing meaning through thick description: verbal representations that accommodate heterogeneity and retain contextual information needed to represent human meaning. While these methods can effectively codify meaning, they are difficult to deploy at scale. However, the verbal capabilities of LLMs now provide a means of (at least partially) automating the generation and processing of thick descriptions, potentially overcoming this bottleneck. We argue that the problem of rendering human meaning legible is not just about selecting better metrics, but about developing new representational formats (based on thick description). We frame this as a crucial direction for the application of generative AI and identify five key challenges: preserving context, maintaining interpretive pluralism, integrating perspectives based on lived experience and critical distance, distinguishing qualitative content from quantitative magnitude, and acknowledging meaning as dynamic rather than static. Furthermore, we suggest that thick description has the potential to serve as a unifying framework to address a number of emerging concerns about the difficulties of representing culture in (or using) LLMs. 

---
# Open CaptchaWorld: A Comprehensive Web-based Platform for Testing and Benchmarking Multimodal LLM Agents 

**Authors**: Yaxin Luo, Zhaoyi Li, Jiacheng Liu, Jiacheng Cui, Xiaohan Zhao, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24878)  

**Abstract**: CAPTCHAs have been a critical bottleneck for deploying web agents in real-world applications, often blocking them from completing end-to-end automation tasks. While modern multimodal LLM agents have demonstrated impressive performance in static perception tasks, their ability to handle interactive, multi-step reasoning challenges like CAPTCHAs is largely untested. To address this gap, we introduce Open CaptchaWorld, the first web-based benchmark and platform specifically designed to evaluate the visual reasoning and interaction capabilities of MLLM-powered agents through diverse and dynamic CAPTCHA puzzles. Our benchmark spans 20 modern CAPTCHA types, totaling 225 CAPTCHAs, annotated with a new metric we propose: CAPTCHA Reasoning Depth, which quantifies the number of cognitive and motor steps required to solve each puzzle. Experimental results show that humans consistently achieve near-perfect scores, state-of-the-art MLLM agents struggle significantly, with success rates at most 40.0% by Browser-Use Openai-o3, far below human-level performance, 93.3%. This highlights Open CaptchaWorld as a vital benchmark for diagnosing the limits of current multimodal agents and guiding the development of more robust multimodal reasoning systems. Code and Data are available at this https URL. 

---
# Agent-X: Evaluating Deep Multimodal Reasoning in Vision-Centric Agentic Tasks 

**Authors**: Tajamul Ashraf, Amal Saqib, Hanan Ghani, Muhra AlMahri, Yuhao Li, Noor Ahsan, Umair Nawaz, Jean Lahoud, Hisham Cholakkal, Mubarak Shah, Philip Torr, Fahad Shahbaz Khan, Rao Muhammad Anwer, Salman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2505.24876)  

**Abstract**: Deep reasoning is fundamental for solving complex tasks, especially in vision-centric scenarios that demand sequential, multimodal understanding. However, existing benchmarks typically evaluate agents with fully synthetic, single-turn queries, limited visual modalities, and lack a framework to assess reasoning quality over multiple steps as required in real-world settings. To address this, we introduce Agent-X, a large-scale benchmark for evaluating vision-centric agents multi-step and deep reasoning capabilities in real-world, multimodal settings. Agent- X features 828 agentic tasks with authentic visual contexts, including images, multi-image comparisons, videos, and instructional text. These tasks span six major agentic environments: general visual reasoning, web browsing, security and surveillance, autonomous driving, sports, and math reasoning. Our benchmark requires agents to integrate tool use with explicit, stepwise decision-making in these diverse settings. In addition, we propose a fine-grained, step-level evaluation framework that assesses the correctness and logical coherence of each reasoning step and the effectiveness of tool usage throughout the task. Our results reveal that even the best-performing models, including GPT, Gemini, and Qwen families, struggle to solve multi-step vision tasks, achieving less than 50% full-chain success. These findings highlight key bottlenecks in current LMM reasoning and tool-use capabilities and identify future research directions in vision-centric agentic reasoning models. Our data and code are publicly available at this https URL 

---
# ReasonGen-R1: CoT for Autoregressive Image generation models through SFT and RL 

**Authors**: Yu Zhang, Yunqi Li, Yifan Yang, Rui Wang, Yuqing Yang, Dai Qi, Jianmin Bao, Dongdong Chen, Chong Luo, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24875)  

**Abstract**: Although chain-of-thought reasoning and reinforcement learning (RL) have driven breakthroughs in NLP, their integration into generative vision models remains underexplored. We introduce ReasonGen-R1, a two-stage framework that first imbues an autoregressive image generator with explicit text-based "thinking" skills via supervised fine-tuning on a newly generated reasoning dataset of written rationales, and then refines its outputs using Group Relative Policy Optimization. To enable the model to reason through text before generating images, We automatically generate and release a corpus of model crafted rationales paired with visual prompts, enabling controlled planning of object layouts, styles, and scene compositions. Our GRPO algorithm uses reward signals from a pretrained vision language model to assess overall visual quality, optimizing the policy in each update. Evaluations on GenEval, DPG, and the T2I benchmark demonstrate that ReasonGen-R1 consistently outperforms strong baselines and prior state-of-the-art models. More: this http URL. 

---
# ProxyThinker: Test-Time Guidance through Small Visual Reasoners 

**Authors**: Zilin Xiao, Jaywon Koo, Siru Ouyang, Jefferson Hernandez, Yu Meng, Vicente Ordonez  

**Link**: [PDF](https://arxiv.org/pdf/2505.24872)  

**Abstract**: Recent advancements in reinforcement learning with verifiable rewards have pushed the boundaries of the visual reasoning capabilities in large vision-language models (LVLMs). However, training LVLMs with reinforcement fine-tuning (RFT) is computationally expensive, posing a significant challenge to scaling model size. In this work, we propose ProxyThinker, an inference-time technique that enables large models to inherit the visual reasoning capabilities from small, slow-thinking visual reasoners without any training. By subtracting the output distributions of base models from those of RFT reasoners, ProxyThinker modifies the decoding dynamics and successfully elicits the slow-thinking reasoning demonstrated by the emerged sophisticated behaviors such as self-verification and self-correction. ProxyThinker consistently boosts performance on challenging visual benchmarks on spatial, mathematical, and multi-disciplinary reasoning, enabling untuned base models to compete with the performance of their full-scale RFT counterparts. Furthermore, our implementation efficiently coordinates multiple language models with parallelism techniques and achieves up to 38 $\times$ faster inference compared to previous decoding-time methods, paving the way for the practical deployment of ProxyThinker. Code is available at this https URL. 

---
# MoDoMoDo: Multi-Domain Data Mixtures for Multimodal LLM Reinforcement Learning 

**Authors**: Yiqing Liang, Jielin Qiu, Wenhao Ding, Zuxin Liu, James Tompkin, Mengdi Xu, Mengzhou Xia, Zhengzhong Tu, Laixi Shi, Jiacheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24871)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a powerful paradigm for post-training large language models (LLMs), achieving state-of-the-art performance on tasks with structured, verifiable answers. Applying RLVR to Multimodal LLMs (MLLMs) presents significant opportunities but is complicated by the broader, heterogeneous nature of vision-language tasks that demand nuanced visual, logical, and spatial capabilities. As such, training MLLMs using RLVR on multiple datasets could be beneficial but creates challenges with conflicting objectives from interaction among diverse datasets, highlighting the need for optimal dataset mixture strategies to improve generalization and reasoning. We introduce a systematic post-training framework for Multimodal LLM RLVR, featuring a rigorous data mixture problem formulation and benchmark implementation. Specifically, (1) We developed a multimodal RLVR framework for multi-dataset post-training by curating a dataset that contains different verifiable vision-language problems and enabling multi-domain online RL learning with different verifiable rewards; (2) We proposed a data mixture strategy that learns to predict the RL fine-tuning outcome from the data mixture distribution, and consequently optimizes the best mixture. Comprehensive experiments showcase that multi-domain RLVR training, when combined with mixture prediction strategies, can significantly boost MLLM general reasoning capacities. Our best mixture improves the post-trained model's accuracy on out-of-distribution benchmarks by an average of 5.24% compared to the same model post-trained with uniform data mixture, and by a total of 20.74% compared to the pre-finetuning baseline. 

---
# Beyond Multiple Choice: Evaluating Steering Vectors for Adaptive Free-Form Summarization 

**Authors**: Joschka Braun, Carsten Eickhoff, Seyed Ali Bahrainian  

**Link**: [PDF](https://arxiv.org/pdf/2505.24859)  

**Abstract**: Steering vectors are a lightweight method for controlling text properties by adding a learned bias to language model activations at inference time. So far, steering vectors have predominantly been evaluated in multiple-choice settings, while their effectiveness in free-form generation tasks remains understudied. Moving "Beyond Multiple Choice," we thoroughly evaluate the effectiveness of steering vectors in adaptively controlling topical focus, sentiment, toxicity, and readability in abstractive summaries of the NEWTS dataset. We find that steering effectively controls the targeted summary properties, but high steering strengths consistently degrade both intrinsic and extrinsic text quality. Compared to steering, prompting offers weaker control, while preserving text quality. Combining steering and prompting yields the strongest control over text properties and offers the most favorable efficacy-quality trade-off at moderate steering strengths. Our results underscore the practical trade-off between control strength and text quality preservation when applying steering vectors to free-form generation tasks. 

---
# Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning 

**Authors**: Shuyao Xu, Cheng Peng, Jiangxuan Long, Weidi Xu, Wei Chu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.24850)  

**Abstract**: Recent advances in model distillation demonstrate that data from advanced reasoning models (e.g., DeepSeek-R1, OpenAI's o1) can effectively transfer complex reasoning abilities to smaller, efficient student models. However, standard practices employ rejection sampling, discarding incorrect reasoning examples -- valuable, yet often underutilized data. This paper addresses the critical question: How can both positive and negative distilled reasoning traces be effectively leveraged to maximize LLM reasoning performance in an offline setting? To this end, We propose Reinforcement Distillation (REDI), a two-stage framework. Stage 1 learns from positive traces via Supervised Fine-Tuning (SFT). Stage 2 further refines the model using both positive and negative traces through our proposed REDI objective. This novel objective is a simple, reference-free loss function that outperforms established methods like DPO and SimPO in this distillation context. Our empirical evaluations demonstrate REDI's superiority over baseline Rejection Sampling SFT or SFT combined with DPO/SimPO on mathematical reasoning tasks. Notably, the Qwen-REDI-1.5B model, post-trained on just 131k positive and negative examples from the open Open-R1 dataset, achieves an 83.1% score on MATH-500 (pass@1). Its performance matches or surpasses that of DeepSeek-R1-Distill-Qwen-1.5B (a model post-trained on 800k proprietary data) across various mathematical reasoning benchmarks, establishing a new state-of-the-art for 1.5B models post-trained offline with openly available data. 

---
# MiCRo: Mixture Modeling and Context-aware Routing for Personalized Preference Learning 

**Authors**: Jingyan Shen, Jiarui Yao, Rui Yang, Yifan Sun, Feng Luo, Rui Pan, Tong Zhang, Han Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.24846)  

**Abstract**: Reward modeling is a key step in building safe foundation models when applying reinforcement learning from human feedback (RLHF) to align Large Language Models (LLMs). However, reward modeling based on the Bradley-Terry (BT) model assumes a global reward function, failing to capture the inherently diverse and heterogeneous human preferences. Hence, such oversimplification limits LLMs from supporting personalization and pluralistic alignment. Theoretically, we show that when human preferences follow a mixture distribution of diverse subgroups, a single BT model has an irreducible error. While existing solutions, such as multi-objective learning with fine-grained annotations, help address this issue, they are costly and constrained by predefined attributes, failing to fully capture the richness of human values. In this work, we introduce MiCRo, a two-stage framework that enhances personalized preference learning by leveraging large-scale binary preference datasets without requiring explicit fine-grained annotations. In the first stage, MiCRo introduces context-aware mixture modeling approach to capture diverse human preferences. In the second stage, MiCRo integrates an online routing strategy that dynamically adapts mixture weights based on specific context to resolve ambiguity, allowing for efficient and scalable preference adaptation with minimal additional supervision. Experiments on multiple preference datasets demonstrate that MiCRo effectively captures diverse human preferences and significantly improves downstream personalization. 

---
# Chameleon: A Flexible Data-mixing Framework for Language Model Pretraining and Finetuning 

**Authors**: Wanyun Xie, Francesco Tonin, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2505.24844)  

**Abstract**: Training data mixtures greatly impact the generalization performance of large language models. Existing domain reweighting methods often rely on costly weight computations and require retraining when new data is introduced. To this end, we introduce a flexible and efficient data mixing framework, Chameleon, that employs leverage scores to quantify domain importance within a learned embedding space. We first construct a domain affinity matrix over domain embeddings. The induced leverage scores determine a mixture that upweights domains sharing common representations in embedding space. This formulation allows direct transfer to new data by computing the new domain embeddings. In experiments, we demonstrate improvements over three key scenarios: (i) our computed weights improve performance on pretraining domains with a fraction of the compute of existing methods; (ii) Chameleon can adapt to data changes without proxy retraining, boosting few-shot reasoning accuracies when transferred to new data; (iii) our method enables efficient domain reweighting in finetuning, consistently improving test perplexity on all finetuning domains over uniform mixture. Our code is available at this https URL. 

---
# Vision LLMs Are Bad at Hierarchical Visual Understanding, and LLMs Are the Bottleneck 

**Authors**: Yuwen Tan, Yuan Qing, Boqing Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.24840)  

**Abstract**: This paper reveals that many state-of-the-art large language models (LLMs) lack hierarchical knowledge about our visual world, unaware of even well-established biology taxonomies. This shortcoming makes LLMs a bottleneck for vision LLMs' hierarchical visual understanding (e.g., recognizing Anemone Fish but not Vertebrate). We arrive at these findings using about one million four-choice visual question answering (VQA) tasks constructed from six taxonomies and four image datasets. Interestingly, finetuning a vision LLM using our VQA tasks reaffirms LLMs' bottleneck effect to some extent because the VQA tasks improve the LLM's hierarchical consistency more than the vision LLM's. We conjecture that one cannot make vision LLMs understand visual concepts fully hierarchical until LLMs possess corresponding taxonomy knowledge. 

---
# PhySense: Principle-Based Physics Reasoning Benchmarking for Large Language Models 

**Authors**: Yinggan Xu, Yue Liu, Zhiqiang Gao, Changnan Peng, Di Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.24823)  

**Abstract**: Large language models (LLMs) have rapidly advanced and are increasingly capable of tackling complex scientific problems, including those in physics. Despite this progress, current LLMs often fail to emulate the concise, principle-based reasoning characteristic of human experts, instead generating lengthy and opaque solutions. This discrepancy highlights a crucial gap in their ability to apply core physical principles for efficient and interpretable problem solving. To systematically investigate this limitation, we introduce PhySense, a novel principle-based physics reasoning benchmark designed to be easily solvable by experts using guiding principles, yet deceptively difficult for LLMs without principle-first reasoning. Our evaluation across multiple state-of-the-art LLMs and prompt types reveals a consistent failure to align with expert-like reasoning paths, providing insights for developing AI systems with efficient, robust and interpretable principle-based scientific reasoning. 

---
# Draw ALL Your Imagine: A Holistic Benchmark and Agent Framework for Complex Instruction-based Image Generation 

**Authors**: Yucheng Zhou, Jiahao Yuan, Qianning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24787)  

**Abstract**: Recent advancements in text-to-image (T2I) generation have enabled models to produce high-quality images from textual descriptions. However, these models often struggle with complex instructions involving multiple objects, attributes, and spatial relationships. Existing benchmarks for evaluating T2I models primarily focus on general text-image alignment and fail to capture the nuanced requirements of complex, multi-faceted prompts. Given this gap, we introduce LongBench-T2I, a comprehensive benchmark specifically designed to evaluate T2I models under complex instructions. LongBench-T2I consists of 500 intricately designed prompts spanning nine diverse visual evaluation dimensions, enabling a thorough assessment of a model's ability to follow complex instructions. Beyond benchmarking, we propose an agent framework (Plan2Gen) that facilitates complex instruction-driven image generation without requiring additional model training. This framework integrates seamlessly with existing T2I models, using large language models to interpret and decompose complex prompts, thereby guiding the generation process more effectively. As existing evaluation metrics, such as CLIPScore, fail to adequately capture the nuances of complex instructions, we introduce an evaluation toolkit that automates the quality assessment of generated images using a set of multi-dimensional metrics. The data and code are released at this https URL. 

---
# REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards 

**Authors**: Zafir Stojanovski, Oliver Stanley, Joe Sharratt, Richard Jones, Abdulhakeem Adefioye, Jean Kaddour, Andreas Köpf  

**Link**: [PDF](https://arxiv.org/pdf/2505.24760)  

**Abstract**: We introduce Reasoning Gym (RG), a library of reasoning environments for reinforcement learning with verifiable rewards. It provides over 100 data generators and verifiers spanning multiple domains including algebra, arithmetic, computation, cognition, geometry, graph theory, logic, and various common games. Its key innovation is the ability to generate virtually infinite training data with adjustable complexity, unlike most previous reasoning datasets, which are typically fixed. This procedural generation approach allows for continuous evaluation across varying difficulty levels. Our experimental results demonstrate the efficacy of RG in both evaluating and reinforcement learning of reasoning models. 

---
# SUMO: Subspace-Aware Moment-Orthogonalization for Accelerating Memory-Efficient LLM Training 

**Authors**: Yehonathan Refael, Guy Smorodinsky, Tom Tirer, Ofir Lindenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2505.24749)  

**Abstract**: Low-rank gradient-based optimization methods have significantly improved memory efficiency during the training of large language models (LLMs), enabling operations within constrained hardware without sacrificing performance. However, these methods primarily emphasize memory savings, often overlooking potential acceleration in convergence due to their reliance on standard isotropic steepest descent techniques, which can perform suboptimally in the highly anisotropic landscapes typical of deep networks, particularly LLMs. In this paper, we propose SUMO (Subspace-Aware Moment-Orthogonalization), an optimizer that employs exact singular value decomposition (SVD) for moment orthogonalization within a dynamically adapted low-dimensional subspace, enabling norm-inducing steepest descent optimization steps. By explicitly aligning optimization steps with the spectral characteristics of the loss landscape, SUMO effectively mitigates approximation errors associated with commonly used methods like Newton-Schulz orthogonalization approximation. We theoretically establish an upper bound on these approximation errors, proving their dependence on the condition numbers of moments, conditions we analytically demonstrate are encountered during LLM training. Furthermore, we both theoretically and empirically illustrate that exact orthogonalization via SVD substantially improves convergence rates while reducing overall complexity. Empirical evaluations confirm that SUMO accelerates convergence, enhances stability, improves performance, and reduces memory requirements by up to 20% compared to state-of-the-art methods. 

---
# "Dyadosyncrasy", Idiosyncrasy and Demographic Factors in Turn-Taking 

**Authors**: Julio Cesar Cavalcanti, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2505.24736)  

**Abstract**: Turn-taking in dialogue follows universal constraints but also varies significantly. This study examines how demographic (sex, age, education) and individual factors shape turn-taking using a large dataset of US English conversations (Fisher). We analyze Transition Floor Offset (TFO) and find notable interspeaker variation. Sex and age have small but significant effects female speakers and older individuals exhibit slightly shorter offsets - while education shows no effect. Lighter topics correlate with shorter TFOs. However, individual differences have a greater impact, driven by a strong idiosyncratic and an even stronger "dyadosyncratic" component - speakers in a dyad resemble each other more than they resemble themselves in different dyads. This suggests that the dyadic relationship and joint activity are the strongest determinants of TFO, outweighing demographic influences. 

---
# CoRet: Improved Retriever for Code Editing 

**Authors**: Fabio Fehr, Prabhu Teja Sivaprasad, Luca Franceschi, Giovanni Zappella  

**Link**: [PDF](https://arxiv.org/pdf/2505.24715)  

**Abstract**: In this paper, we introduce CoRet, a dense retrieval model designed for code-editing tasks that integrates code semantics, repository structure, and call graph dependencies. The model focuses on retrieving relevant portions of a code repository based on natural language queries such as requests to implement new features or fix bugs. These retrieved code chunks can then be presented to a user or to a second code-editing model or agent. To train CoRet, we propose a loss function explicitly designed for repository-level retrieval. On SWE-bench and Long Code Arena's bug localisation datasets, we show that our model substantially improves retrieval recall by at least 15 percentage points over existing models, and ablate the design choices to show their importance in achieving these results. 

---
# Causal-aware Large Language Models: Enhancing Decision-Making Through Learning, Adapting and Acting 

**Authors**: Wei Chen, Jiahao Zhang, Haipeng Zhu, Boyan Xu, Zhifeng Hao, Keli Zhang, Junjian Ye, Ruichu Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.24710)  

**Abstract**: Large language models (LLMs) have shown great potential in decision-making due to the vast amount of knowledge stored within the models. However, these pre-trained models are prone to lack reasoning abilities and are difficult to adapt to new environments, further hindering their application to complex real-world tasks. To address these challenges, inspired by the human cognitive process, we propose Causal-aware LLMs, which integrate the structural causal model (SCM) into the decision-making process to model, update, and utilize structured knowledge of the environment in a ``learning-adapting-acting" paradigm. Specifically, in the learning stage, we first utilize an LLM to extract the environment-specific causal entities and their causal relations to initialize a structured causal model of the environment. Subsequently,in the adapting stage, we update the structured causal model through external feedback about the environment, via an idea of causal intervention. Finally, in the acting stage, Causal-aware LLMs exploit structured causal knowledge for more efficient policy-making through the reinforcement learning agent. The above processes are performed iteratively to learn causal knowledge, ultimately enabling the causal-aware LLMs to achieve a more accurate understanding of the environment and make more efficient decisions. Experimental results across 22 diverse tasks within the open-world game ``Crafter" validate the effectiveness of our proposed method. 

---
# Identifying Primary Stress Across Related Languages and Dialects with Transformer-based Speech Encoder Models 

**Authors**: Nikola Ljubešić, Ivan Porupski, Peter Rupnik  

**Link**: [PDF](https://arxiv.org/pdf/2505.24571)  

**Abstract**: Automating primary stress identification has been an active research field due to the role of stress in encoding meaning and aiding speech comprehension. Previous studies relied mainly on traditional acoustic features and English datasets. In this paper, we investigate the approach of fine-tuning a pre-trained transformer model with an audio frame classification head. Our experiments use a new Croatian training dataset, with test sets in Croatian, Serbian, the Chakavian dialect, and Slovenian. By comparing an SVM classifier using traditional acoustic features with the fine-tuned speech transformer, we demonstrate the transformer's superiority across the board, achieving near-perfect results for Croatian and Serbian, with a 10-point performance drop for the more distant Chakavian and Slovenian. Finally, we show that only a few hundred multi-syllabic training words suffice for strong performance. We release our datasets and model under permissive licenses. 

---
# Beyond Linear Steering: Unified Multi-Attribute Control for Language Models 

**Authors**: Narmeen Oozeer, Luke Marks, Fazl Barez, Amirali Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2505.24535)  

**Abstract**: Controlling multiple behavioral attributes in large language models (LLMs) at inference time is a challenging problem due to interference between attributes and the limitations of linear steering methods, which assume additive behavior in activation space and require per-attribute tuning. We introduce K-Steering, a unified and flexible approach that trains a single non-linear multi-label classifier on hidden activations and computes intervention directions via gradients at inference time. This avoids linearity assumptions, removes the need for storing and tuning separate attribute vectors, and allows dynamic composition of behaviors without retraining. To evaluate our method, we propose two new benchmarks, ToneBank and DebateMix, targeting compositional behavioral control. Empirical results across 3 model families, validated by both activation-based classifiers and LLM-based judges, demonstrate that K-Steering outperforms strong baselines in accurately steering multiple behaviors. 

---
# AMIA: Automatic Masking and Joint Intention Analysis Makes LVLMs Robust Jailbreak Defenders 

**Authors**: Yuqi Zhang, Yuchun Miao, Zuchao Li, Liang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.24519)  

**Abstract**: We introduce AMIA, a lightweight, inference-only defense for Large Vision-Language Models (LVLMs) that (1) Automatically Masks a small set of text-irrelevant image patches to disrupt adversarial perturbations, and (2) conducts joint Intention Analysis to uncover and mitigate hidden harmful intents before response generation. Without any retraining, AMIA improves defense success rates across diverse LVLMs and jailbreak benchmarks from an average of 52.4% to 81.7%, preserves general utility with only a 2% average accuracy drop, and incurs only modest inference overhead. Ablation confirms both masking and intention analysis are essential for a robust safety-utility trade-off. 

---
# Leveraging Knowledge Graphs and LLMs for Structured Generation of Misinformation 

**Authors**: Sania Nayab, Marco Simoni, Giulio Rossolini  

**Link**: [PDF](https://arxiv.org/pdf/2505.24479)  

**Abstract**: The rapid spread of misinformation, further amplified by recent advances in generative AI, poses significant threats to society, impacting public opinion, democratic stability, and national security. Understanding and proactively assessing these threats requires exploring methodologies that enable structured and scalable misinformation generation. In this paper, we propose a novel approach that leverages knowledge graphs (KGs) as structured semantic resources to systematically generate fake triplets. By analyzing the structural properties of KGs, such as the distance between entities and their predicates, we identify plausibly false relationships. These triplets are then used to guide large language models (LLMs) in generating misinformation statements with varying degrees of credibility. By utilizing structured semantic relationships, our deterministic approach produces misinformation inherently challenging for humans to detect, drawing exclusively upon publicly available KGs (e.g., WikiGraphs).
Additionally, we investigate the effectiveness of LLMs in distinguishing between genuine and artificially generated misinformation. Our analysis highlights significant limitations in current LLM-based detection methods, underscoring the necessity for enhanced detection strategies and a deeper exploration of inherent biases in generative models. 

---
# Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning 

**Authors**: Vasilije Markovic, Lazar Obradovic, Laszlo Hajdu, Jovan Pavlovic  

**Link**: [PDF](https://arxiv.org/pdf/2505.24478)  

**Abstract**: Integrating Large Language Models (LLMs) with Knowledge Graphs (KGs) results in complex systems with numerous hyperparameters that directly affect performance. While such systems are increasingly common in retrieval-augmented generation, the role of systematic hyperparameter optimization remains underexplored. In this paper, we study this problem in the context of Cognee, a modular framework for end-to-end KG construction and retrieval. Using three multi-hop QA benchmarks (HotPotQA, TwoWikiMultiHop, and MuSiQue) we optimize parameters related to chunking, graph construction, retrieval, and prompting. Each configuration is scored using established metrics (exact match, F1, and DeepEval's LLM-based correctness metric). Our results demonstrate that meaningful gains can be achieved through targeted tuning. While the gains are consistent, they are not uniform, with performance varying across datasets and metrics. This variability highlights both the value of tuning and the limitations of standard evaluation measures. While demonstrating the immediate potential of hyperparameter tuning, we argue that future progress will depend not only on architectural advances but also on clearer frameworks for optimization and evaluation in complex, modular systems. 

---
# Breaking the Gold Standard: Extracting Forgotten Data under Exact Unlearning in Large Language Models 

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24379)  

**Abstract**: Large language models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard, believed to be robust against privacy-related attacks. In this paper, we challenge this assumption by introducing a novel data extraction attack that compromises even exact unlearning. Our method leverages both the pre- and post-unlearning models: by guiding the post-unlearning model using signals from the pre-unlearning model, we uncover patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. 

---
# KEVER^2: Knowledge-Enhanced Visual Emotion Reasoning and Retrieval 

**Authors**: Fanhang Man, Xiaoyue Chen, Huandong Wang, Baining Zhao, Han Li, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24342)  

**Abstract**: Understanding what emotions images evoke in their viewers is a foundational goal in human-centric visual computing. While recent advances in vision-language models (VLMs) have shown promise for visual emotion analysis (VEA), several key challenges remain unresolved. Emotional cues in images are often abstract, overlapping, and entangled, making them difficult to model and interpret. Moreover, VLMs struggle to align these complex visual patterns with emotional semantics due to limited supervision and sparse emotional grounding. Finally, existing approaches lack structured affective knowledge to resolve ambiguity and ensure consistent emotional reasoning across diverse visual domains.
To address these limitations, we propose \textbf{K-EVER\textsuperscript{2}}, a knowledge-enhanced framework for emotion reasoning and retrieval. Our approach introduces a semantically structured formulation of visual emotion cues and integrates external affective knowledge through multimodal alignment. Without relying on handcrafted labels or direct emotion supervision, K-EVER\textsuperscript{2} achieves robust and interpretable emotion predictions across heterogeneous image types.
We validate our framework on three representative benchmarks, Emotion6, EmoSet, and M-Disaster, covering social media imagery, human-centric scenes, and disaster contexts. K-EVER\textsuperscript{2} consistently outperforms strong CNN and VLM baselines, achieving up to a \textbf{19\% accuracy gain} for specific emotions and a \textbf{12.3\% average accuracy gain} across all emotion categories. Our results demonstrate a scalable and generalizable solution for advancing emotional understanding of visual content. 

---
# GeoVision Labeler: Zero-Shot Geospatial Classification with Vision and Language Models 

**Authors**: Gilles Quentin Hacheme, Girmaw Abebe Tadesse, Caleb Robinson, Akram Zaytar, Rahul Dodhia, Juan M. Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2505.24340)  

**Abstract**: Classifying geospatial imagery remains a major bottleneck for applications such as disaster response and land-use monitoring-particularly in regions where annotated data is scarce or unavailable. Existing tools (e.g., RS-CLIP) that claim zero-shot classification capabilities for satellite imagery nonetheless rely on task-specific pretraining and adaptation to reach competitive performance. We introduce GeoVision Labeler (GVL), a strictly zero-shot classification framework: a vision Large Language Model (vLLM) generates rich, human-readable image descriptions, which are then mapped to user-defined classes by a conventional Large Language Model (LLM). This modular, and interpretable pipeline enables flexible image classification for a large range of use cases. We evaluated GVL across three benchmarks-SpaceNet v7, UC Merced, and RESISC45. It achieves up to 93.2% zero-shot accuracy on the binary Buildings vs. No Buildings task on SpaceNet v7. For complex multi-class classification tasks (UC Merced, RESISC45), we implemented a recursive LLM-driven clustering to form meta-classes at successive depths, followed by hierarchical classification-first resolving coarse groups, then finer distinctions-to deliver competitive zero-shot performance. GVL is open-sourced at this https URL to catalyze adoption in real-world geospatial workflows. 

---
# SwiftEval: Developing a Language-Specific Benchmark for LLM-generated Code Evaluation 

**Authors**: Ivan Petrukha, Yana Kurliak, Nataliia Stulova  

**Link**: [PDF](https://arxiv.org/pdf/2505.24324)  

**Abstract**: In recent years, large language models (LLMs) have showcased significant advancements in code generation. However, most evaluation benchmarks are primarily oriented towards Python, making it difficult to evaluate other programming languages, such as Swift, with high quality. By examining widely established multilingual benchmarks like HumanEval-XL and MultiPL-E, we identified critical issues specific to their Swift components, making them insufficient or even irrelevant for assessing LLM coding capabilities on Swift. Unlike these existing approaches, which prioritize rapid scaling and generalization by automatically translating Python-centric benchmarks with LLMs, we adopt a quality-over-quantity methodology. We present SwiftEval, the first Swift-oriented benchmark consisting of 28 carefully hand-crafted problems, and evaluate 44 popular Code LLMs on it. Our results show significant LLM scores drop for problems requiring language-specific features, most noticeable in the models of smaller sizes. 

---
# Large Language Models are Locally Linear Mappings 

**Authors**: James R. Golden  

**Link**: [PDF](https://arxiv.org/pdf/2505.24293)  

**Abstract**: We demonstrate that the inference operations of several open-weight large language models (LLMs) can be mapped to an exactly equivalent linear system for an input sequence without modifying the model weights or altering output predictions. Extending techniques from image diffusion models that exhibit local or piecewise linearity, we strategically alter the gradient computation with respect to a given input sequence for a next-token prediction such that the Jacobian of the model nearly exactly reproduces the forward prediction with a linear system. We demonstrate this approach across models (Llama 3, Gemma 3, Qwen 3, Phi 4, Mistral Ministral and OLMo 2, up to Llama 3.3 70B Q4) and show through the singular value decomposition of the detached Jacobian that these LLMs operate in extremely low-dimensional subspaces where many of the largest singular vectors decode to concepts related to the most-likely output token. This approach also allows us to examine the operation of each successive layer (and its attention and MLP components) as nearly-exact linear systems and observe the emergence of semantic concepts. Despite their expressive power and global nonlinearity, modern LLMs can be interpreted through nearly-exact locally linear decompositions that provide insights into their internal representations and reveal interpretable semantic structures in the next-token prediction process. 

---
# Mind the Quote: Enabling Quotation-Aware Dialogue in LLMs via Plug-and-Play Modules 

**Authors**: Yueqi Zhang, Peiwen Yuan, Shaoxiong Feng, Yiwei Li, Xinglin Wang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24292)  

**Abstract**: Human-AI conversation frequently relies on quoting earlier text-"check it with the formula I just highlighted"-yet today's large language models (LLMs) lack an explicit mechanism for locating and exploiting such spans. We formalise the challenge as span-conditioned generation, decomposing each turn into the dialogue history, a set of token-offset quotation spans, and an intent utterance. Building on this abstraction, we introduce a quotation-centric data pipeline that automatically synthesises task-specific dialogues, verifies answer correctness through multi-stage consistency checks, and yields both a heterogeneous training corpus and the first benchmark covering five representative scenarios. To meet the benchmark's zero-overhead and parameter-efficiency requirements, we propose QuAda, a lightweight training-based method that attaches two bottleneck projections to every attention head, dynamically amplifying or suppressing attention to quoted spans at inference time while leaving the prompt unchanged and updating < 2.8% of backbone weights. Experiments across models show that QuAda is suitable for all scenarios and generalises to unseen topics, offering an effective, plug-and-play solution for quotation-aware dialogue. 

---
# An Adversary-Resistant Multi-Agent LLM System via Credibility Scoring 

**Authors**: Sana Ebrahimi, Mohsen Dehghankar, Abolfazl Asudeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.24239)  

**Abstract**: While multi-agent LLM systems show strong capabilities in various domains, they are highly vulnerable to adversarial and low-performing agents. To resolve this issue, in this paper, we introduce a general and adversary-resistant multi-agent LLM framework based on credibility scoring. We model the collaborative query-answering process as an iterative game, where the agents communicate and contribute to a final system output. Our system associates a credibility score that is used when aggregating the team outputs. The credibility scores are learned gradually based on the past contributions of each agent in query answering. Our experiments across multiple tasks and settings demonstrate our system's effectiveness in mitigating adversarial influence and enhancing the resilience of multi-agent cooperation, even in the adversary-majority settings. 

---
# From Hallucinations to Jailbreaks: Rethinking the Vulnerability of Large Foundation Models 

**Authors**: Haibo Jin, Peiyan Zhang, Peiran Wang, Man Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24232)  

**Abstract**: Large foundation models (LFMs) are susceptible to two distinct vulnerabilities: hallucinations and jailbreak attacks. While typically studied in isolation, we observe that defenses targeting one often affect the other, hinting at a deeper connection.
We propose a unified theoretical framework that models jailbreaks as token-level optimization and hallucinations as attention-level optimization. Within this framework, we establish two key propositions: (1) \textit{Similar Loss Convergence} - the loss functions for both vulnerabilities converge similarly when optimizing for target-specific outputs; and (2) \textit{Gradient Consistency in Attention Redistribution} - both exhibit consistent gradient behavior driven by shared attention dynamics.
We validate these propositions empirically on LLaVA-1.5 and MiniGPT-4, showing consistent optimization trends and aligned gradients. Leveraging this connection, we demonstrate that mitigation techniques for hallucinations can reduce jailbreak success rates, and vice versa. Our findings reveal a shared failure mode in LFMs and suggest that robustness strategies should jointly address both vulnerabilities. 

---
# Reasoning Can Hurt the Inductive Abilities of Large Language Models 

**Authors**: Haibo Jin, Peiyan Zhang, Man Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24225)  

**Abstract**: Large Language Models (LLMs) have shown remarkable progress across domains, yet their ability to perform inductive reasoning - inferring latent rules from sparse examples - remains limited. It is often assumed that chain-of-thought (CoT) prompting, as used in Large Reasoning Models (LRMs), enhances such reasoning. We investigate this assumption with creating four controlled, diagnostic game-based tasks - chess, Texas Hold'em, dice games, and blackjack - with hidden human-defined rules. We find that CoT reasoning can degrade inductive performance, with LRMs often underperforming their non-reasoning counterparts.
To explain this, we present a theoretical framework that reveals how reasoning steps can amplify error through three failure modes: incorrect sub-task decomposition, incorrect sub-task solving, and incorrect final answer summarization. Based on our theoretical and empirical analysis, we introduce structured interventions that adapt CoT generation according to our identified failure types. These interventions improve inductive accuracy without retraining. Our findings suggest that effective (CoT) reasoning depends not only on taking more steps but also on ensuring those steps are well-structured. 

---
# Improving Multilingual Speech Models on ML-SUPERB 2.0: Fine-tuning with Data Augmentation and LID-Aware CTC 

**Authors**: Qingzheng Wang, Jiancheng Sun, Yifan Peng, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2505.24200)  

**Abstract**: Multilingual speech processing with self-supervised or supervised pre-trained Speech Foundation Models (SFM) has achieved strong performance on tasks like Language Identification (LID) and Automatic Speech Recognition (ASR). However, these models struggle with limited resources during fine-tuning. This paper enhances multilingual LID and ASR on ML-SUPERB 2.0 by exploring multiple strategies for adapting SFMs, including frozen upstream training, partial fine-tuning, and low-rank adaptation. Furthermore, we employ data augmentation to mitigate performance gaps in few-shot settings and introduce LID Connectionist Temporal Classification (CTC) loss for regularization. Our approach achieves a 14% relative improvement in LID accuracy and a 30% relative reduction in ASR CER over the baseline on ML-SUPERB 2.0, securing second place in the Interspeech 2025 ML-SUPERB 2.0 Challenge. 

---
# WikiGap: Promoting Epistemic Equity by Surfacing Knowledge Gaps Between English Wikipedia and other Language Editions 

**Authors**: Zining Wang, Yuxuan Zhang, Dongwook Yoon, Nicholas Vincent, Farhan Samir, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2505.24195)  

**Abstract**: With more than 11 times as many pageviews as the next, English Wikipedia dominates global knowledge access relative to other language editions. Readers are prone to assuming English Wikipedia as a superset of all language editions, leading many to prefer it even when their primary language is not English. Other language editions, however, comprise complementary facts rooted in their respective cultures and media environments, which are marginalized in English Wikipedia. While Wikipedia's user interface enables switching between language editions through its Interlanguage Link (ILL) system, it does not reveal to readers that other language editions contain valuable, complementary information. We present WikiGap, a system that surfaces complementary facts sourced from other Wikipedias within the English Wikipedia interface. Specifically, by combining a recent multilingual information-gap discovery method with a user-centered design, WikiGap enables access to complementary information from French, Russian, and Chinese Wikipedia. In a mixed-methods study (n=21), WikiGap significantly improved fact-finding accuracy, reduced task time, and received a 32-point higher usability score relative to Wikipedia's current ILL-based navigation system. Participants reported increased awareness of the availability of complementary information in non-English editions and reconsidered the completeness of English Wikipedia. WikiGap thus paves the way for improved epistemic equity across language editions. 

---
# Fine-Tune an SLM or Prompt an LLM? The Case of Generating Low-Code Workflows 

**Authors**: Orlando Marquez Ayala, Patrice Bechard, Emily Chen, Maggie Baird, Jingfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24189)  

**Abstract**: Large Language Models (LLMs) such as GPT-4o can handle a wide range of complex tasks with the right prompt. As per token costs are reduced, the advantages of fine-tuning Small Language Models (SLMs) for real-world applications -- faster inference, lower costs -- may no longer be clear. In this work, we present evidence that, for domain-specific tasks that require structured outputs, SLMs still have a quality advantage. We compare fine-tuning an SLM against prompting LLMs on the task of generating low-code workflows in JSON form. We observe that while a good prompt can yield reasonable results, fine-tuning improves quality by 10% on average. We also perform systematic error analysis to reveal model limitations. 

---
# Redefining Research Crowdsourcing: Incorporating Human Feedback with LLM-Powered Digital Twins 

**Authors**: Amanda Chan, Catherine Di, Joseph Rupertus, Gary Smith, Varun Nagaraj Rao, Manoel Horta Ribeiro, Andrés Monroy-Hernández  

**Link**: [PDF](https://arxiv.org/pdf/2505.24004)  

**Abstract**: Crowd work platforms like Amazon Mechanical Turk and Prolific are vital for research, yet workers' growing use of generative AI tools poses challenges. Researchers face compromised data validity as AI responses replace authentic human behavior, while workers risk diminished roles as AI automates tasks. To address this, we propose a hybrid framework using digital twins, personalized AI models that emulate workers' behaviors and preferences while keeping humans in the loop. We evaluate our system with an experiment (n=88 crowd workers) and in-depth interviews with crowd workers (n=5) and social science researchers (n=4). Our results suggest that digital twins may enhance productivity and reduce decision fatigue while maintaining response quality. Both researchers and workers emphasized the importance of transparency, ethical data use, and worker agency. By automating repetitive tasks and preserving human engagement for nuanced ones, digital twins may help balance scalability with authenticity. 

---
# Large Language Models for Controllable Multi-property Multi-objective Molecule Optimization 

**Authors**: Vishal Dey, Xiao Hu, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2505.23987)  

**Abstract**: In real-world drug design, molecule optimization requires selectively improving multiple molecular properties up to pharmaceutically relevant levels, while maintaining others that already meet such criteria. However, existing computational approaches and instruction-tuned LLMs fail to capture such nuanced property-specific objectives, limiting their practical applicability. To address this, we introduce C-MuMOInstruct, the first instruction-tuning dataset focused on multi-property optimization with explicit, property-specific objectives. Leveraging C-MuMOInstruct, we develop GeLLMO-Cs, a series of instruction-tuned LLMs that can perform targeted property-specific optimization. Our experiments across 5 in-distribution and 5 out-of-distribution tasks show that GeLLMO-Cs consistently outperform strong baselines, achieving up to 126% higher success rate. Notably, GeLLMO-Cs exhibit impressive 0-shot generalization to novel optimization tasks and unseen instructions. This offers a step toward a foundational LLM to support realistic, diverse optimizations with property-specific objectives. C-MuMOInstruct and code are accessible through this https URL. 

---
# Information Structure in Mappings: An Approach to Learning, Representation, and Generalisation 

**Authors**: Henry Conklin  

**Link**: [PDF](https://arxiv.org/pdf/2505.23960)  

**Abstract**: Despite the remarkable success of large large-scale neural networks, we still lack unified notation for thinking about and describing their representational spaces. We lack methods to reliably describe how their representations are structured, how that structure emerges over training, and what kinds of structures are desirable. This thesis introduces quantitative methods for identifying systematic structure in a mapping between spaces, and leverages them to understand how deep-learning models learn to represent information, what representational structures drive generalisation, and how design decisions condition the structures that emerge. To do this I identify structural primitives present in a mapping, along with information theoretic quantifications of each. These allow us to analyse learning, structure, and generalisation across multi-agent reinforcement learning models, sequence-to-sequence models trained on a single task, and Large Language Models. I also introduce a novel, performant, approach to estimating the entropy of vector space, that allows this analysis to be applied to models ranging in size from 1 million to 12 billion parameters.
The experiments here work to shed light on how large-scale distributed models of cognition learn, while allowing us to draw parallels between those systems and their human analogs. They show how the structures of language and the constraints that give rise to them in many ways parallel the kinds of structures that drive performance of contemporary neural networks. 

---
# ScaleLong: A Multi-Timescale Benchmark for Long Video Understanding 

**Authors**: David Ma, Huaqing Yuan, Xingjian Wang, Qianbo Zang, Tianci Liu, Xinyang He, Yanbin Wei, Jiawei Guo, Ni Jiahui, Zhenzhu Yang, Meng Cao, Shanghaoran Quan, Yizhi Li, Wangchunshu Zhou, Jiaheng Liu, Wenhao Huang, Ge Zhang, Shiwen Ni, Xiaojie Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.23922)  

**Abstract**: Although long-video understanding demands that models capture hierarchical temporal information -- from clip (seconds) and shot (tens of seconds) to event (minutes) and story (hours) -- existing benchmarks either neglect this multi-scale design or scatter scale-specific questions across different videos, preventing direct comparison of model performance across timescales on the same content. To address this, we introduce ScaleLong, the first benchmark to disentangle these factors by embedding questions targeting four hierarchical timescales -- clip (seconds), shot (tens of seconds), event (minutes), and story (hours) -- all within the same video content. This within-content multi-timescale questioning design enables direct comparison of model performance across timescales on identical videos. ScaleLong features 269 long videos (avg.\ 86\,min) from 5 main categories and 36 sub-categories, with 4--8 carefully designed questions, including at least one question for each timescale. Evaluating 23 MLLMs reveals a U-shaped performance curve, with higher accuracy at the shortest and longest timescales and a dip at intermediate levels. Furthermore, ablation studies show that increased visual token capacity consistently enhances reasoning across all timescales. ScaleLong offers a fine-grained, multi-timescale benchmark for advancing MLLM capabilities in long-video understanding. The code and dataset are available this https URL. 

---
# OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation 

**Authors**: Mengkang Hu, Yuhang Zhou, Wendong Fan, Yuzhou Nie, Bowei Xia, Tao Sun, Ziyu Ye, Zhaoxuan Jin, Yingru Li, Qiguang Chen, Zeyu Zhang, Yifeng Wang, Qianshuo Ye, Bernard Ghanem, Ping Luo, Guohao Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23885)  

**Abstract**: Large Language Model (LLM)-based multi-agent systems show promise for automating real-world tasks but struggle to transfer across domains due to their domain-specific nature. Current approaches face two critical shortcomings: they require complete architectural redesign and full retraining of all components when applied to new domains. We introduce Workforce, a hierarchical multi-agent framework that decouples strategic planning from specialized execution through a modular architecture comprising: (i) a domain-agnostic Planner for task decomposition, (ii) a Coordinator for subtask management, and (iii) specialized Workers with domain-specific tool-calling capabilities. This decoupling enables cross-domain transferability during both inference and training phases: During inference, Workforce seamlessly adapts to new domains by adding or modifying worker agents; For training, we introduce Optimized Workforce Learning (OWL), which improves generalization across domains by optimizing a domain-agnostic planner with reinforcement learning from real-world feedback. To validate our approach, we evaluate Workforce on the GAIA benchmark, covering various realistic, multi-domain agentic tasks. Experimental results demonstrate Workforce achieves open-source state-of-the-art performance (69.70%), outperforming commercial systems like OpenAI's Deep Research by 2.34%. More notably, our OWL-trained 32B model achieves 52.73% accuracy (+16.37%) and demonstrates performance comparable to GPT-4o on challenging tasks. To summarize, by enabling scalable generalization and modular domain transfer, our work establishes a foundation for the next generation of general-purpose AI assistants. 

---
# Test-Time Training Done Right 

**Authors**: Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T. Freeman, Hao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.23884)  

**Abstract**: Test-Time Training (TTT) models context dependencies by adapting part of the model's weights (referred to as fast weights) during inference. This fast weight, akin to recurrent states in RNNs, stores temporary memories of past tokens in the current sequence. Existing TTT methods struggled to show effectiveness in handling long-context data, due to their inefficiency on modern GPUs. The TTT layers in many of these approaches operate with extremely low FLOPs utilization (often <5%) because they deliberately apply small online minibatch sizes (e.g., updating fast weights every 16 or 64 tokens). Moreover, a small minibatch implies fine-grained block-wise causal dependencies in the data, unsuitable for data beyond 1D ordered sequences, like sets or N-dimensional grids such as images or videos. In contrast, we pursue the opposite direction by using an extremely large chunk update, ranging from 2K to 1M tokens across tasks of varying modalities, which we refer to as Large Chunk Test-Time Training (LaCT). It improves hardware utilization by orders of magnitude, and more importantly, facilitates scaling of nonlinear state size (up to 40% of model parameters), hence substantially improving state capacity, all without requiring cumbersome and error-prone kernel implementations. It also allows easy integration of sophisticated optimizers, e.g. Muon for online updates. We validate our approach across diverse modalities and tasks, including novel view synthesis with image set, language models, and auto-regressive video diffusion. Our approach can scale up to 14B-parameter AR video diffusion model on sequences up to 56K tokens. In our longest sequence experiment, we perform novel view synthesis with 1 million context length. We hope this work will inspire and accelerate new research in the field of long-context modeling and test-time training. Website: this https URL 

---
# BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning 

**Authors**: Jianyang Gu, Samuel Stevens, Elizabeth G Campolongo, Matthew J Thompson, Net Zhang, Jiaman Wu, Andrei Kopanev, Zheda Mai, Alexander E. White, James Balhoff, Wasila Dahdul, Daniel Rubenstein, Hilmar Lapp, Tanya Berger-Wolf, Wei-Lun Chao, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.23883)  

**Abstract**: Foundation models trained at scale exhibit remarkable emergent behaviors, learning new capabilities beyond their initial training objectives. We find such emergent behaviors in biological vision models via large-scale contrastive vision-language training. To achieve this, we first curate TreeOfLife-200M, comprising 214 million images of living organisms, the largest and most diverse biological organism image dataset to date. We then train BioCLIP 2 on TreeOfLife-200M to distinguish different species. Despite the narrow training objective, BioCLIP 2 yields extraordinary accuracy when applied to various biological visual tasks such as habitat classification and trait prediction. We identify emergent properties in the learned embedding space of BioCLIP 2. At the inter-species level, the embedding distribution of different species aligns closely with functional and ecological meanings (e.g., beak sizes and habitats). At the intra-species level, instead of being diminished, the intra-species variations (e.g., life stages and sexes) are preserved and better separated in subspaces orthogonal to inter-species distinctions. We provide formal proof and analyses to explain why hierarchical supervision and contrastive objectives encourage these emergent properties. Crucially, our results reveal that these properties become increasingly significant with larger-scale training data, leading to a biologically meaningful embedding space. 

---
# Using Reasoning Models to Generate Search Heuristics that Solve Open Instances of Combinatorial Design Problems 

**Authors**: Christopher D. Rosin  

**Link**: [PDF](https://arxiv.org/pdf/2505.23881)  

**Abstract**: Large Language Models (LLMs) with reasoning are trained to iteratively generate and refine their answers before finalizing them, which can help with applications to mathematics and code generation. We apply code generation with reasoning LLMs to a specific task in the mathematical field of combinatorial design. This field studies diverse types of combinatorial designs, many of which have lists of open instances for which existence has not yet been determined. The Constructive Protocol CPro1 uses LLMs to generate search heuristics that have the potential to construct solutions to small open instances. Starting with a textual definition and a validity verifier for a particular type of design, CPro1 guides LLMs to select and implement strategies, while providing automated hyperparameter tuning and execution feedback. CPro1 with reasoning LLMs successfully solves long-standing open instances for 7 of 16 combinatorial design problems selected from the 2006 Handbook of Combinatorial Designs, including new solved instances for 3 of these (Bhaskar Rao Designs, Symmetric Weighing Matrices, Balanced Ternary Designs) that were unsolved by CPro1 with non-reasoning LLMs. It also solves open instances for several problems from recent (2025) literature, generating new Covering Sequences, Johnson Clique Covers, Deletion Codes, and a Uniform Nested Steiner Quadruple System. 

---
# SkewRoute: Training-Free LLM Routing for Knowledge Graph Retrieval-Augmented Generation via Score Skewness of Retrieved Context 

**Authors**: Hairu Wang, Yuan Feng, Yukun Cao, Xike Xie, S Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.23841)  

**Abstract**: Large language models excel at many tasks but often incur high inference costs during deployment. To mitigate hallucination, many systems use a knowledge graph to enhance retrieval-augmented generation (KG-RAG). However, the large amount of retrieved knowledge contexts increase these inference costs further. A promising solution to balance performance and cost is LLM routing, which directs simple queries to smaller LLMs and complex ones to larger LLMs. However, no dedicated routing methods currently exist for RAG, and existing training-based routers face challenges scaling to this domain due to the need for extensive training data. We observe that the score distributions produced by the retrieval scorer strongly correlate with query difficulty. Based on this, we propose a novel, training-free routing framework, the first tailored to KG-RAG that effectively balances performance and cost in a plug-and-play manner. Experiments show our method reduces calls to larger LLMs by up to 50% without sacrificing response quality, demonstrating its potential for efficient and scalable LLM deployment. 

---
# Boosting In-Context Learning in LLMs Through the Lens of Classical Supervised Learning 

**Authors**: Korel Gundem, Juncheng Dong, Dennis Zhang, Vahid Tarokh, Zhengling Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.23783)  

**Abstract**: In-Context Learning (ICL) allows Large Language Models (LLMs) to adapt to new tasks with just a few examples, but their predictions often suffer from systematic biases, leading to unstable performances in classification. While calibration techniques are proposed to mitigate these biases, we show that, in the logit space, many of these methods are equivalent to merely shifting the LLM's decision boundary without having the ability to alter its orientation. This proves inadequate when biases cause the LLM to be severely misdirected. To address these limitations and provide a unifying framework, we propose Supervised Calibration (SC), a loss-minimization based framework which learns an optimal, per-class affine transformation of the LLM's predictive probabilities in the logit space without requiring external data beyond the context. By using a more expressive functional class, SC not only subsumes many existing calibration methods in ICL as special cases, but also enables the ability to alter and even completely reverse the orientation of the LLM's decision boundary. Furthermore, SC's loss-based nature facilitates the seamless integration of two purpose-built regularization techniques: context-invariance and directional trust-region. The former is designed to tackle the instability issue in ICL, while the latter controls the degree of calibration. Finally, SC delivers state-of-the-art performance over calibration baselines in the 4-shot, 8-shot, and 16-shot settings across all nine datasets for Mistral-7B-Instruct-v0.3, LLaMA-2-7B-chat, and Qwen2-7B-Instruct. 

---
