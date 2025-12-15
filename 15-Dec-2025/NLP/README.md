# SUMFORU: An LLM-Based Review Summarization Framework for Personalized Purchase Decision Support 

**Authors**: Yuming Feng, Xinrui Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.11755)  

**Abstract**: Online product reviews contain rich but noisy signals that overwhelm users and hinder effective decision-making. Existing LLM-based summarizers remain generic and fail to account for individual preferences, limiting their practical utility. We propose SUMFORU, a steerable review summarization framework that aligns outputs with explicit user personas to support personalized purchase decisions. Our approach integrates a high-quality data pipeline built from the Amazon 2023 Review Dataset with a two-stage alignment procedure: (1) persona-aware Supervised Fine-Tuning (SFT) via asymmetric knowledge distillation, and (2) Reinforcement Learning with AI Feedback (RLAIF) using a preference estimator to capture fine-grained, persona-relevant signals. We evaluate the model across rule-based, LLM-based, and human-centered metrics, demonstrating consistent improvements in consistency, grounding, and preference alignment. Our framework achieves the highest performance across all evaluation settings and generalizes effectively to unseen product categories. Our results highlight the promise of steerable pluralistic alignment for building next-generation personalized decision-support systems. 

---
# Speculative Decoding Speed-of-Light: Optimal Lower Bounds via Branching Random Walks 

**Authors**: Sergey Pankratov, Dan Alistarh  

**Link**: [PDF](https://arxiv.org/pdf/2512.11718)  

**Abstract**: Speculative generation has emerged as a promising technique to accelerate inference in large language models (LLMs) by leveraging parallelism to verify multiple draft tokens simultaneously. However, the fundamental limits on the achievable speedup remain poorly understood. In this work, we establish the first ``tight'' lower bounds on the runtime of any deterministic speculative generation algorithm. This is achieved by drawing a parallel between the token generation process and branching random walks, which allows us to analyze the optimal draft tree selection problem. We prove, under basic assumptions, that the expected number of tokens successfully predicted per speculative iteration is bounded as $\mathbb{E}[X] \leq (\mu + \mu_{(2)})\log(P )/\mu^2 + O(1)$, where $P$ is the verifier's capacity, $\mu$ is the expected entropy of the verifier's output distribution, and $\mu_{(2)}$ is the expected second log-moment. This result provides new insights into the limits of parallel token generation, and could guide the design of future speculative decoding systems. Empirical evaluations on Llama models validate our theoretical predictions, confirming the tightness of our bounds in practical settings. 

---
# Automating Historical Insight Extraction from Large-Scale Newspaper Archives via Neural Topic Modeling 

**Authors**: Keerthana Murugaraj, Salima Lamsiyah, Marten During, Martin Theobald  

**Link**: [PDF](https://arxiv.org/pdf/2512.11635)  

**Abstract**: Extracting coherent and human-understandable themes from large collections of unstructured historical newspaper archives presents significant challenges due to topic evolution, Optical Character Recognition (OCR) noise, and the sheer volume of text. Traditional topic-modeling methods, such as Latent Dirichlet Allocation (LDA), often fall short in capturing the complexity and dynamic nature of discourse in historical texts. To address these limitations, we employ BERTopic. This neural topic-modeling approach leverages transformerbased embeddings to extract and classify topics, which, despite its growing popularity, still remains underused in historical research. Our study focuses on articles published between 1955 and 2018, specifically examining discourse on nuclear power and nuclear safety. We analyze various topic distributions across the corpus and trace their temporal evolution to uncover long-term trends and shifts in public discourse. This enables us to more accurately explore patterns in public discourse, including the co-occurrence of themes related to nuclear power and nuclear weapons and their shifts in topic importance over time. Our study demonstrates the scalability and contextual sensitivity of BERTopic as an alternative to traditional approaches, offering richer insights into historical discourses extracted from newspaper archives. These findings contribute to historical, nuclear, and social-science research while reflecting on current limitations and proposing potential directions for future work. 

---
# Bounding Hallucinations: Information-Theoretic Guarantees for RAG Systems via Merlin-Arthur Protocols 

**Authors**: Björn Deiseroth, Max Henning Höth, Kristian Kersting, Letitia Parcalabescu  

**Link**: [PDF](https://arxiv.org/pdf/2512.11614)  

**Abstract**: Retrieval-augmented generation (RAG) models rely on retrieved evidence to guide large language model (LLM) generators, yet current systems treat retrieval as a weak heuristic rather than verifiable evidence. As a result, LLMs answer without support, hallucinate under incomplete or misleading context, and rely on spurious evidence. We introduce a training framework that treats the entire RAG pipeline -- both the retriever and the generator -- as an interactive proof system via an adaptation of the Merlin-Arthur (M/A) protocol. Arthur (the generator LLM) trains on questions of unkown provenance: Merlin provides helpful evidence, while Morgana injects adversarial, misleading context. Both use a linear-time XAI method to identify and modify the evidence most influential to Arthur. Consequently, Arthur learns to (i) answer when the context support the answer, (ii) reject when evidence is insufficient, and (iii) rely on the specific context spans that truly ground the answer. We further introduce a rigorous evaluation framework to disentangle explanation fidelity from baseline predictive errors. This allows us to introduce and measure the Explained Information Fraction (EIF), which normalizes M/A certified mutual-information guarantees relative to model capacity and imperfect benchmarks. Across three RAG datasets and two model families of varying sizes, M/A-trained LLMs show improved groundedness, completeness, soundness, and reject behavior, as well as reduced hallucinations -- without needing manually annotated unanswerable questions. The retriever likewise improves recall and MRR through automatically generated M/A hard positives and negatives. Our results demonstrate that autonomous interactive-proof-style supervision provides a principled and practical path toward reliable RAG systems that treat retrieved documents not as suggestions, but as verifiable evidence. 

---
# Visualizing token importance for black-box language models 

**Authors**: Paulius Rauba, Qiyao Wei, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2512.11573)  

**Abstract**: We consider the problem of auditing black-box large language models (LLMs) to ensure they behave reliably when deployed in production settings, particularly in high-stakes domains such as legal, medical, and regulatory compliance. Existing approaches for LLM auditing often focus on isolated aspects of model behavior, such as detecting specific biases or evaluating fairness. We are interested in a more general question -- can we understand how the outputs of black-box LLMs depend on each input token? There is a critical need to have such tools in real-world applications that rely on inaccessible API endpoints to language models. However, this is a highly non-trivial problem, as LLMs are stochastic functions (i.e. two outputs will be different by chance), while computing prompt-level gradients to approximate input sensitivity is infeasible. To address this, we propose Distribution-Based Sensitivity Analysis (DBSA), a lightweight model-agnostic procedure to evaluate the sensitivity of the output of a language model for each input token, without making any distributional assumptions about the LLM. DBSA is developed as a practical tool for practitioners, enabling quick, plug-and-play visual exploration of LLMs reliance on specific input tokens. Through illustrative examples, we demonstrate how DBSA can enable users to inspect LLM inputs and find sensitivities that may be overlooked by existing LLM interpretability methods. 

---
# Extending a Parliamentary Corpus with MPs' Tweets: Automatic Annotation and Evaluation Using MultiParTweet 

**Authors**: Mevlüt Bagci, Ali Abusaleh, Daniel Baumartz, Giueseppe Abrami, Maxim Konca, Alexander Mehler  

**Link**: [PDF](https://arxiv.org/pdf/2512.11567)  

**Abstract**: Social media serves as a critical medium in modern politics because it both reflects politicians' ideologies and facilitates communication with younger generations. We present MultiParTweet, a multilingual tweet corpus from X that connects politicians' social media discourse with German political corpus GerParCor, thereby enabling comparative analyses between online communication and parliamentary debates. MultiParTweet contains 39 546 tweets, including 19 056 media items. Furthermore, we enriched the annotation with nine text-based models and one vision-language model (VLM) to annotate MultiParTweet with emotion, sentiment, and topic annotations. Moreover, the automated annotations are evaluated against a manually annotated subset. MultiParTweet can be reconstructed using our tool, TTLABTweetCrawler, which provides a framework for collecting data from X. To demonstrate a methodological demonstration, we examine whether the models can predict each other using the outputs of the remaining models. In summary, we provide MultiParTweet, a resource integrating automatic text and media-based annotations validated with human annotations, and TTLABTweetCrawler, a general-purpose X data collection tool. Our analysis shows that the models are mutually predictable. In addition, VLM-based annotation were preferred by human annotators, suggesting that multimodal representations align more with human interpretation. 

---
# Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs 

**Authors**: Mohor Banerjee, Nadya Yuki Wangsajaya, Syed Ali Redha Alsagoff, Min Sen Tan, Zachary Choy Kit Chun, Alvin Chan Guo Wei  

**Link**: [PDF](https://arxiv.org/pdf/2512.11509)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities in natural language understanding and reasoning, but suffer from hallucination: the generation of factually incorrect content. While numerous methods have been developed to reduce hallucinations, their impact on creative generations remains unexplored. This gap is particularly critical for AI-assisted scientific discovery, which requires both factual accuracy and creative hypothesis generation. We investigate how three hallucination-reduction techniques: Chain of Verification (CoVe), Decoding by Contrasting Layers (DoLa), and Retrieval-Augmented Generation (RAG), affect creativity in LLMs. Evaluating multiple model families (LLaMA, Qwen, Mistral) at varying scales (1B - 70B parameters) on two creativity benchmarks (NeoCoder and CS4), we find that these methods have opposing effects on divergent creativity. CoVe enhances divergent thinking, DoLa suppresses it, and RAG shows minimal impact. Our findings provide guidance for selecting appropriate hallucination-reduction methods in scientific applications, where the balance between factual accuracy and creative exploration is crucial. 

---
# Building Patient Journeys in Hebrew: A Language Model for Clinical Timeline Extraction 

**Authors**: Kai Golan Hashiloni, Brenda Kasabe Nokai, Michal Shevach, Esthy Shemesh, Ronit Bartin, Anna Bergrin, Liran Harel, Nachum Dershowitz, Liat Nadai Arad, Kfir Bar  

**Link**: [PDF](https://arxiv.org/pdf/2512.11502)  

**Abstract**: We present a new Hebrew medical language model designed to extract structured clinical timelines from electronic health records, enabling the construction of patient journeys. Our model is based on DictaBERT 2.0 and continually pre-trained on over five million de-identified hospital records. To evaluate its effectiveness, we introduce two new datasets -- one from internal medicine and emergency departments, and another from oncology -- annotated for event temporal relations. Our results show that our model achieves strong performance on both datasets. We also find that vocabulary adaptation improves token efficiency and that de-identification does not compromise downstream performance, supporting privacy-conscious model development. The model is made available for research use under ethical restrictions. 

---
# Mistake Notebook Learning: Selective Batch-Wise Context Optimization for In-Context Learning 

**Authors**: Xuanbo Su, Yingfang Zhang, Hao Luo, Xiaoteng Liu, Leo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.11485)  

**Abstract**: Large language models (LLMs) adapt to tasks via gradient fine-tuning (heavy computation, catastrophic forgetting) or In-Context Learning (ICL: low robustness, poor mistake learning). To fix this, we introduce Mistake Notebook Learning (MNL), a training-free framework with a persistent knowledge base of abstracted error patterns. Unlike prior instance/single-trajectory memory methods, MNL uses batch-wise error abstraction: it extracts generalizable guidance from multiple failures, stores insights in a dynamic notebook, and retains only baseline-outperforming guidance via hold-out validation (ensuring monotonic improvement). We show MNL nearly matches Supervised Fine-Tuning (93.9% vs 94.3% on GSM8K) and outperforms training-free alternatives on GSM8K, Spider, AIME, and KaggleDBQA. On KaggleDBQA (Qwen3-8B), MNL hits 28% accuracy (47% relative gain), outperforming Memento (15.1%) and Training-Free GRPO (22.1) - proving it's a strong training-free alternative for complex reasoning. 

---
# CLINIC: Evaluating Multilingual Trustworthiness in Language Models for Healthcare 

**Authors**: Akash Ghosh, Srivarshinee Sridhar, Raghav Kaushik Ravi, Muhsin Muhsin, Sriparna Saha, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2512.11437)  

**Abstract**: Integrating language models (LMs) in healthcare systems holds great promise for improving medical workflows and decision-making. However, a critical barrier to their real-world adoption is the lack of reliable evaluation of their trustworthiness, especially in multilingual healthcare settings. Existing LMs are predominantly trained in high-resource languages, making them ill-equipped to handle the complexity and diversity of healthcare queries in mid- and low-resource languages, posing significant challenges for deploying them in global healthcare contexts where linguistic diversity is key. In this work, we present CLINIC, a Comprehensive Multilingual Benchmark to evaluate the trustworthiness of language models in healthcare. CLINIC systematically benchmarks LMs across five key dimensions of trustworthiness: truthfulness, fairness, safety, robustness, and privacy, operationalized through 18 diverse tasks, spanning 15 languages (covering all the major continents), and encompassing a wide array of critical healthcare topics like disease conditions, preventive actions, diagnostic tests, treatments, surgeries, and medications. Our extensive evaluation reveals that LMs struggle with factual correctness, demonstrate bias across demographic and linguistic groups, and are susceptible to privacy breaches and adversarial attacks. By highlighting these shortcomings, CLINIC lays the foundation for enhancing the global reach and safety of LMs in healthcare across diverse languages. 

---
# Minimal Clips, Maximum Salience: Long Video Summarization via Key Moment Extraction 

**Authors**: Galann Pennec, Zhengyuan Liu, Nicholas Asher, Philippe Muller, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.11399)  

**Abstract**: Vision-Language Models (VLMs) are able to process increasingly longer videos. Yet, important visual information is easily lost throughout the entire context and missed by VLMs. Also, it is important to design tools that enable cost-effective analysis of lengthy video content. In this paper, we propose a clip selection method that targets key video moments to be included in a multimodal summary. We divide the video into short clips and generate compact visual descriptions of each using a lightweight video captioning model. These are then passed to a large language model (LLM), which selects the K clips containing the most relevant visual information for a multimodal summary. We evaluate our approach on reference clips for the task, automatically derived from full human-annotated screenplays and summaries in the MovieSum dataset. We further show that these reference clips (less than 6% of the movie) are sufficient to build a complete multimodal summary of the movies in MovieSum. Using our clip selection method, we achieve a summarization performance close to that of these reference clips while capturing substantially more relevant video information than random clip selection. Importantly, we maintain low computational cost by relying on a lightweight captioning model. 

---
# Improving Translation Quality by Selecting Better Data for LLM Fine-Tuning: A Comparative Analysis 

**Authors**: Felipe Ribeiro Fujita de Mello, Hideyuki Takada  

**Link**: [PDF](https://arxiv.org/pdf/2512.11388)  

**Abstract**: We investigated the impact of data selection on machine translation fine-tuning for open LLMs. Using Japanese-English corpora, we compare five selectors: TF-IDF, COMET Kiwi, QuRate, FD-Score, and random selection, under controlled training conditions. We observed that semantic selectors consistently outperform lexical and geometry-based heuristics, and that even when the selected data differ by less than 3%, the impact on model performance is substantial, underscoring the sensitivity of fine-tuning to data quality. 

---
# Mining Legal Arguments to Study Judicial Formalism 

**Authors**: Tomáš Koref, Lena Held, Mahammad Namazov, Harun Kumru, Yassine Thlija, Christoph Burchard, Ivan Habernal  

**Link**: [PDF](https://arxiv.org/pdf/2512.11374)  

**Abstract**: Courts must justify their decisions, but systematically analyzing judicial reasoning at scale remains difficult. This study refutes claims about formalistic judging in Central and Eastern Europe (CEE) by developing automated methods to detect and classify judicial reasoning in Czech Supreme Courts' decisions using state-of-the-art natural language processing methods. We create the MADON dataset of 272 decisions from two Czech Supreme Courts with expert annotations of 9,183 paragraphs with eight argument types and holistic formalism labels for supervised training and evaluation. Using a corpus of 300k Czech court decisions, we adapt transformer LLMs for Czech legal domain by continued pretraining and experiment with methods to address dataset imbalance including asymmetric loss and class weighting. The best models successfully detect argumentative paragraphs (82.6\% macro-F1), classify traditional types of legal argument (77.5\% macro-F1), and classify decisions as formalistic/non-formalistic (83.2\% macro-F1). Our three-stage pipeline combining ModernBERT, Llama 3.1, and traditional feature-based machine learning achieves promising results for decision classification while reducing computational costs and increasing explainability. Empirically, we challenge prevailing narratives about CEE formalism. This work shows that legal argument mining enables reliable judicial philosophy classification and shows the potential of legal argument mining for other important tasks in computational legal studies. Our methodology is easily replicable across jurisdictions, and our entire pipeline, datasets, guidelines, models, and source codes are available at this https URL. 

---
# qa-FLoRA: Data-free query-adaptive Fusion of LoRAs for LLMs 

**Authors**: Shreya Shukla, Aditya Sriram, Milinda Kuppur Narayanaswamy, Hiteshi Jain  

**Link**: [PDF](https://arxiv.org/pdf/2512.11366)  

**Abstract**: The deployment of large language models for specialized tasks often requires domain-specific parameter-efficient finetuning through Low-Rank Adaptation (LoRA) modules. However, effectively fusing these adapters to handle complex, multi-domain composite queries remains a critical challenge. Existing LoRA fusion approaches either use static weights, which assign equal relevance to each participating LoRA, or require data-intensive supervised training for every possible LoRA combination to obtain respective optimal fusion weights. We propose qa-FLoRA, a novel query-adaptive data-and-training-free method for LoRA fusion that dynamically computes layer-level fusion weights by measuring distributional divergence between the base model and respective adapters. Our approach eliminates the need for composite training data or domain-representative samples, making it readily applicable to existing adapter collections. Extensive experiments across nine multilingual composite tasks spanning mathematics, coding, and medical domains, show that qa-FLoRA outperforms static fusion by ~5% with LLaMA-2 and ~6% with LLaMA-3, and the training-free baselines by ~7% with LLaMA-2 and ~10% with LLaMA-3, while significantly closing the gap with supervised baselines. Further, layer-level analysis of our fusion weights reveals interpretable fusion patterns, demonstrating the effectiveness of our approach for robust multi-domain adaptation. 

---
# Unifying Dynamic Tool Creation and Cross-Task Experience Sharing through Cognitive Memory Architecture 

**Authors**: Jiarun Liu, Shiyue Xu, Yang Li, Shangkun Liu, Yongli Yu, Peng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2512.11303)  

**Abstract**: Large Language Model agents face fundamental challenges in adapting to novel tasks due to limitations in tool availability and experience reuse. Existing approaches either rely on predefined tools with limited coverage or build tools from scratch without leveraging past experiences, leading to inefficient exploration and suboptimal performance. We introduce SMITH (Shared Memory Integrated Tool Hub), a unified cognitive architecture that seamlessly integrates dynamic tool creation with cross-task experience sharing through hierarchical memory organization. SMITH organizes agent memory into procedural, semantic, and episodic components, enabling systematic capability expansion while preserving successful execution patterns. Our approach formalizes tool creation as iterative code generation within controlled sandbox environments and experience sharing through episodic memory retrieval with semantic similarity matching. We further propose a curriculum learning strategy based on agent-ensemble difficulty re-estimation. Extensive experiments on the GAIA benchmark demonstrate SMITH's effectiveness, achieving 81.8% Pass@1 accuracy and outperforming state-of-the-art baselines including Alita (75.2%) and Memento (70.9%). Our work establishes a foundation for building truly adaptive agents that continuously evolve their capabilities through principled integration of tool creation and experience accumulation. 

---
# LegalRikai: Open Benchmark -- A Benchmark for Complex Japanese Corporate Legal Tasks 

**Authors**: Shogo Fujita, Yuji Naraki, Yiqing Zhu, Shinsuke Mori  

**Link**: [PDF](https://arxiv.org/pdf/2512.11297)  

**Abstract**: This paper introduces LegalRikai: Open Benchmark, a new benchmark comprising four complex tasks that emulate Japanese corporate legal practices. The benchmark was created by legal professionals under the supervision of an attorney. This benchmark has 100 samples that require long-form, structured outputs, and we evaluated them against multiple practical criteria. We conducted both human and automated evaluations using leading LLMs, including GPT-5, Gemini 2.5 Pro, and Claude Opus 4.1. Our human evaluation revealed that abstract instructions prompted unnecessary modifications, highlighting model weaknesses in document-level editing that were missed by conventional short-text tasks. Furthermore, our analysis reveals that automated evaluation aligns well with human judgment on criteria with clear linguistic grounding, and assessing structural consistency remains a challenge. The result demonstrates the utility of automated evaluation as a screening tool when expert availability is limited. We propose a dataset evaluation framework to promote more practice-oriented research in the legal domain. 

---
# CIP: A Plug-and-Play Causal Prompting Framework for Mitigating Hallucinations under Long-Context Noise 

**Authors**: Qingsen Ma, Dianyun Wang, Ran Jing, Yujun Sun, Zhenbo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2512.11282)  

**Abstract**: Large language models often hallucinate when processing long and noisy retrieval contexts because they rely on spurious correlations rather than genuine causal relationships. We propose CIP, a lightweight and plug-and-play causal prompting framework that mitigates hallucinations at the input stage. CIP constructs a causal relation sequence among entities, actions, and events and injects it into the prompt to guide reasoning toward causally relevant evidence. Through causal intervention and counterfactual reasoning, CIP suppresses non causal reasoning paths, improving factual grounding and interpretability. Experiments across seven mainstream language models, including GPT-4o, Gemini 2.0 Flash, and Llama 3.1, show that CIP consistently enhances reasoning quality and reliability, achieving 2.6 points improvement in Attributable Rate, 0.38 improvement in Causal Consistency Score, and a fourfold increase in effective information density. API level profiling further shows that CIP accelerates contextual understanding and reduces end to end response latency by up to 55.1 percent. These results suggest that causal reasoning may serve as a promising paradigm for improving the explainability, stability, and efficiency of large language models. 

---
# AdaSD: Adaptive Speculative Decoding for Efficient Language Model Inference 

**Authors**: Kuan-Wei Lu, Ding-Yong Hong, Pangfeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.11280)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across a wide range of tasks, but their increasing parameter sizes significantly slow down inference. Speculative decoding mitigates this issue by leveraging a smaller draft model to predict candidate tokens, which are then verified by a larger target model. However, existing approaches often require additional training, extensive hyperparameter tuning, or prior analysis of models and tasks before deployment. In this paper, we propose Adaptive Speculative Decoding (AdaSD), a hyperparameter-free decoding scheme that dynamically adjusts generation length and acceptance criteria during inference. AdaSD introduces two adaptive thresholds: one to determine when to stop candidate token generation and another to decide token acceptance, both updated in real time based on token entropy and Jensen-Shannon distance. This approach eliminates the need for pre-analysis or fine-tuning and is compatible with off-the-shelf models. Experiments on benchmark datasets demonstrate that AdaSD achieves up to 49\% speedup over standard speculative decoding while limiting accuracy degradation to under 2\%, making it a practical solution for efficient and adaptive LLM inference. 

---
# When Actions Teach You to Think: Reasoning-Action Synergy via Reinforcement Learning in Conversational Agents 

**Authors**: Mrinal Rawat, Arkajyoti Chakraborty, Neha Gupta, Roberto Pieraccini  

**Link**: [PDF](https://arxiv.org/pdf/2512.11277)  

**Abstract**: Supervised fine-tuning (SFT) has emerged as one of the most effective ways to improve the performance of large language models (LLMs) in downstream tasks. However, SFT can have difficulty generalizing when the underlying data distribution changes, even when the new data does not fall completely outside the training domain. Recent reasoning-focused models such as o1 and R1 have demonstrated consistent gains over their non-reasoning counterparts, highlighting the importance of reasoning for improved generalization and reliability. However, collecting high-quality reasoning traces for SFT remains challenging -- annotations are costly, subjective, and difficult to scale. To address this limitation, we leverage Reinforcement Learning (RL) to enable models to learn reasoning strategies directly from task outcomes. We propose a pipeline in which LLMs generate reasoning steps that guide both the invocation of tools (e.g., function calls) and the final answer generation for conversational agents. Our method employs Group Relative Policy Optimization (GRPO) with rewards designed around tool accuracy and answer correctness, allowing the model to iteratively refine its reasoning and actions. Experimental results demonstrate that our approach improves both the quality of reasoning and the precision of tool invocations, achieving a 1.5% relative improvement over the SFT model (trained without explicit thinking) and a 40% gain compared to the base of the vanilla Qwen3-1.7B model. These findings demonstrate the promise of unifying reasoning and action learning through RL to build more capable and generalizable conversational agents. 

---
# Leveraging LLMs for Title and Abstract Screening for Systematic Review: A Cost-Effective Dynamic Few-Shot Learning Approach 

**Authors**: Yun-Chung Liu, Rui Yang, Jonathan Chong Kai Liew, Ziran Yin, Henry Foote, Christopher J. Lindsell, Chuan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2512.11261)  

**Abstract**: Systematic reviews are a key component of evidence-based medicine, playing a critical role in synthesizing existing research evidence and guiding clinical decisions. However, with the rapid growth of research publications, conducting systematic reviews has become increasingly burdensome, with title and abstract screening being one of the most time-consuming and resource-intensive steps. To mitigate this issue, we designed a two-stage dynamic few-shot learning (DFSL) approach aimed at improving the efficiency and performance of large language models (LLMs) in the title and abstract screening task. Specifically, this approach first uses a low-cost LLM for initial screening, then re-evaluates low-confidence instances using a high-performance LLM, thereby enhancing screening performance while controlling computational costs. We evaluated this approach across 10 systematic reviews, and the results demonstrate its strong generalizability and cost-effectiveness, with potential to reduce manual screening burden and accelerate the systematic review process in practical applications. 

---
# Multi-Intent Spoken Language Understanding: Methods, Trends, and Challenges 

**Authors**: Di Wu, Ruiyu Fang, Liting Jiang, Shuangyong Song, Xiaomeng Huang, Shiquan Wang, Zhongqiu Li, Lingling Shi, Mengjiao Bao, Yongxiang Li, Hao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.11258)  

**Abstract**: Multi-intent spoken language understanding (SLU) involves two tasks: multiple intent detection and slot filling, which jointly handle utterances containing more than one intent. Owing to this characteristic, which closely reflects real-world applications, the task has attracted increasing research attention, and substantial progress has been achieved. However, there remains a lack of a comprehensive and systematic review of existing studies on multi-intent SLU. To this end, this paper presents a survey of recent advances in multi-intent SLU. We provide an in-depth overview of previous research from two perspectives: decoding paradigms and modeling approaches. On this basis, we further compare the performance of representative models and analyze their strengths and limitations. Finally, we discuss the current challenges and outline promising directions for future research. We hope this survey will offer valuable insights and serve as a useful reference for advancing research in multi-intent SLU. 

---
# SciLaD: A Large-Scale, Transparent, Reproducible Dataset for Natural Scientific Language Processing 

**Authors**: Luca Foppiano, Sotaro Takeshita, Pedro Ortiz Suarez, Ekaterina Borisova, Raia Abu Ahmad, Malte Ostendorff, Fabio Barth, Julian Moreno-Schneider, Georg Rehm  

**Link**: [PDF](https://arxiv.org/pdf/2512.11192)  

**Abstract**: SciLaD is a novel, large-scale dataset of scientific language constructed entirely using open-source frameworks and publicly available data sources. It comprises a curated English split containing over 10 million scientific publications and a multilingual, unfiltered TEI XML split including more than 35 million publications. We also publish the extensible pipeline for generating SciLaD. The dataset construction and processing workflow demonstrates how open-source tools can enable large-scale, scientific data curation while maintaining high data quality. Finally, we pre-train a RoBERTa model on our dataset and evaluate it across a comprehensive set of benchmarks, achieving performance comparable to other scientific language models of similar size, validating the quality and utility of SciLaD. We publish the dataset and evaluation pipeline to promote reproducibility, transparency, and further research in natural scientific language processing and understanding including scholarly document processing. 

---
# FIBER: A Multilingual Evaluation Resource for Factual Inference Bias 

**Authors**: Evren Ayberk Munis, Deniz Yılmaz, Arianna Muti, Çağrı Toraman  

**Link**: [PDF](https://arxiv.org/pdf/2512.11110)  

**Abstract**: Large language models are widely used across domains, yet there are concerns about their factual reliability and biases. Factual knowledge probing offers a systematic means to evaluate these aspects. Most existing benchmarks focus on single-entity facts and monolingual data. We therefore present FIBER, a multilingual benchmark for evaluating factual knowledge in single- and multi-entity settings. The dataset includes sentence completion, question-answering, and object-count prediction tasks in English, Italian, and Turkish. Using FIBER, we examine whether the prompt language induces inference bias in entity selection and how large language models perform on multi-entity versus single-entity questions. The results indicate that the language of the prompt can influence the model's generated output, particularly for entities associated with the country corresponding to that language. However, this effect varies across different topics such that 31% of the topics exhibit factual inference bias score greater than 0.5. Moreover, the level of bias differs across languages such that Turkish prompts show higher bias compared to Italian in 83% of the topics, suggesting a language-dependent pattern. Our findings also show that models face greater difficulty when handling multi-entity questions than the single-entity questions. Model performance differs across both languages and model sizes. The highest mean average precision is achieved in English, while Turkish and Italian lead to noticeably lower scores. Larger models, including Llama-3.1-8B and Qwen-2.5-7B, show consistently better performance than smaller 3B-4B models. 

---
# Explanation Bias is a Product: Revealing the Hidden Lexical and Position Preferences in Post-Hoc Feature Attribution 

**Authors**: Jonathan Kamp, Roos Bakker, Dominique Blok  

**Link**: [PDF](https://arxiv.org/pdf/2512.11108)  

**Abstract**: Good quality explanations strengthen the understanding of language models and data. Feature attribution methods, such as Integrated Gradient, are a type of post-hoc explainer that can provide token-level insights. However, explanations on the same input may vary greatly due to underlying biases of different methods. Users may be aware of this issue and mistrust their utility, while unaware users may trust them inadequately. In this work, we delve beyond the superficial inconsistencies between attribution methods, structuring their biases through a model- and method-agnostic framework of three evaluation metrics. We systematically assess both the lexical and position bias (what and where in the input) for two transformers; first, in a controlled, pseudo-random classification task on artificial data; then, in a semi-controlled causal relation detection task on natural data. We find that lexical and position biases are structurally unbalanced in our model comparison, with models that score high on one type score low on the other. We also find signs that methods producing anomalous explanations are more likely to be biased themselves. 

---
# Applying NLP to iMessages: Understanding Topic Avoidance, Responsiveness, and Sentiment 

**Authors**: Alan Gerber, Sam Cooperman  

**Link**: [PDF](https://arxiv.org/pdf/2512.11079)  

**Abstract**: What is your messaging data used for? While many users do not often think about the information companies can gather based off of their messaging platform of choice, it is nonetheless important to consider as society increasingly relies on short-form electronic communication. While most companies keep their data closely guarded, inaccessible to users or potential hackers, Apple has opened a door to their walled-garden ecosystem, providing iMessage users on Mac with one file storing all their messages and attached metadata. With knowledge of this locally stored file, the question now becomes: What can our data do for us? In the creation of our iMessage text message analyzer, we set out to answer five main research questions focusing on topic modeling, response times, reluctance scoring, and sentiment analysis. This paper uses our exploratory data to show how these questions can be answered using our analyzer and its potential in future studies on iMessage data. 

---
# MultiScript30k: Leveraging Multilingual Embeddings to Extend Cross Script Parallel Data 

**Authors**: Christopher Driggers-Ellis, Detravious Brinkley, Ray Chen, Aashish Dhawan, Daisy Zhe Wang, Christan Grant  

**Link**: [PDF](https://arxiv.org/pdf/2512.11074)  

**Abstract**: Multi30k is frequently cited in the multimodal machine translation (MMT) literature, offering parallel text data for training and fine-tuning deep learning models. However, it is limited to four languages: Czech, English, French, and German. This restriction has led many researchers to focus their investigations only on these languages. As a result, MMT research on diverse languages has been stalled because the official Multi30k dataset only represents European languages in Latin scripts. Previous efforts to extend Multi30k exist, but the list of supported languages, represented language families, and scripts is still very short. To address these issues, we propose MultiScript30k, a new Multi30k dataset extension for global languages in various scripts, created by translating the English version of Multi30k (Multi30k-En) using NLLB200-3.3B. The dataset consists of over \(30000\) sentences and provides translations of all sentences in Multi30k-En into Ar, Es, Uk, Zh\_Hans and Zh\_Hant. Similarity analysis shows that Multi30k extension consistently achieves greater than \(0.8\) cosine similarity and symmetric KL divergence less than \(0.000251\) for all languages supported except Zh\_Hant which is comparable to the previous Multi30k extensions ArEnMulti30k and Multi30k-Uk. COMETKiwi scores reveal mixed assessments of MultiScript30k as a translation of Multi30k-En in comparison to the related work. ArEnMulti30k scores nearly equal MultiScript30k-Ar, but Multi30k-Uk scores $6.4\%$ greater than MultiScript30k-Uk per split. 

---
# PIAST: Rapid Prompting with In-context Augmentation for Scarce Training data 

**Authors**: Pawel Batorski, Paul Swoboda  

**Link**: [PDF](https://arxiv.org/pdf/2512.11013)  

**Abstract**: LLMs are highly sensitive to prompt design, but handcrafting effective prompts is difficult and often requires intricate crafting of few-shot examples. We propose a fast automatic prompt construction algorithm that augments human instructions by generating a small set of few shot examples. Our method iteratively replaces/drops/keeps few-shot examples using Monte Carlo Shapley estimation of example utility. For faster execution, we use aggressive subsampling and a replay buffer for faster evaluations. Our method can be run using different compute time budgets. On a limited budget, we outperform existing automatic prompting methods on text simplification and GSM8K and obtain second best results on classification and summarization. With an extended, but still modest compute budget we set a new state of the art among automatic prompting methods on classification, simplification and GSM8K. Our results show that carefully constructed examples, rather than exhaustive instruction search, are the dominant lever for fast and data efficient prompt engineering. Our code is available at this https URL. 

---
# KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering 

**Authors**: Xin Sun, Zhongqi Chen, Xing Zheng, Qiang Liu, Shu Wu, Bowen Song, Zilei Wang, Weiqiang Wang, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.10999)  

**Abstract**: Knowledge Base Question Answering (KBQA) challenges models to bridge the gap between natural language and strict knowledge graph schemas by generating executable logical forms. While Large Language Models (LLMs) have advanced this field, current approaches often struggle with a dichotomy of failure: they either generate hallucinated queries without verifying schema existence or exhibit rigid, template-based reasoning that mimics synthesized traces without true comprehension of the environment. To address these limitations, we present \textbf{KBQA-R1}, a framework that shifts the paradigm from text imitation to interaction optimization via Reinforcement Learning. Treating KBQA as a multi-turn decision process, our model learns to navigate the knowledge base using a list of actions, leveraging Group Relative Policy Optimization (GRPO) to refine its strategies based on concrete execution feedback rather than static supervision. Furthermore, we introduce \textbf{Referenced Rejection Sampling (RRS)}, a data synthesis method that resolves cold-start challenges by strictly aligning reasoning traces with ground-truth action sequences. Extensive experiments on WebQSP, GrailQA, and GraphQuestions demonstrate that KBQA-R1 achieves state-of-the-art performance, effectively grounding LLM reasoning in verifiable execution. 

---
# MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA 

**Authors**: Seonok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.10996)  

**Abstract**: Recent advancements in retrieval-augmented generation (RAG) have significantly enhanced the ability of large language models (LLMs) to perform complex question-answering (QA) tasks. In this paper, we introduce MedBioRAG, a retrieval-augmented model designed to improve biomedical QA performance through a combination of semantic and lexical search, document retrieval, and supervised fine-tuning. MedBioRAG efficiently retrieves and ranks relevant biomedical documents, enabling precise and context-aware response generation. We evaluate MedBioRAG across text retrieval, close-ended QA, and long-form QA tasks using benchmark datasets such as NFCorpus, TREC-COVID, MedQA, PubMedQA, and BioASQ. Experimental results demonstrate that MedBioRAG outperforms previous state-of-the-art (SoTA) models and the GPT-4o base model in all evaluated tasks. Notably, our approach improves NDCG and MRR scores for document retrieval, while achieving higher accuracy in close-ended QA and ROUGE scores in long-form QA. Our findings highlight the effectiveness of semantic search-based retrieval and LLM fine-tuning in biomedical applications. 

---
# Benchmarking Automatic Speech Recognition Models for African Languages 

**Authors**: Alvin Nahabwe, Sulaiman Kagumire, Denis Musinguzi, Bruno Beijuka, Jonah Mubuuke Kyagaba, Peter Nabende, Andrew Katumba, Joyce Nakatumba-Nabende  

**Link**: [PDF](https://arxiv.org/pdf/2512.10968)  

**Abstract**: Automatic speech recognition (ASR) for African languages remains constrained by limited labeled data and the lack of systematic guidance on model selection, data scaling, and decoding strategies. Large pre-trained systems such as Whisper, XLS-R, MMS, and W2v-BERT have expanded access to ASR technology, but their comparative behavior in African low-resource contexts has not been studied in a unified and systematic way. In this work, we benchmark four state-of-the-art ASR models across 13 African languages, fine-tuning them on progressively larger subsets of transcribed data ranging from 1 to 400 hours. Beyond reporting error rates, we provide new insights into why models behave differently under varying conditions. We show that MMS and W2v-BERT are more data efficient in very low-resource regimes, XLS-R scales more effectively as additional data becomes available, and Whisper demonstrates advantages in mid-resource conditions. We also analyze where external language model decoding yields improvements and identify cases where it plateaus or introduces additional errors, depending on the alignment between acoustic and text resources. By highlighting the interaction between pre-training coverage, model architecture, dataset domain, and resource availability, this study offers practical and insights into the design of ASR systems for underrepresented languages. 

---
# ASR Under the Stethoscope: Evaluating Biases in Clinical Speech Recognition across Indian Languages 

**Authors**: Subham Kumar, Prakrithi Shivaprakash, Abhishek Manoharan, Astut Kurariya, Diptadhi Mukherjee, Lekhansh Shukla, Animesh Mukherjee, Prabhat Chand, Pratima Murthy  

**Link**: [PDF](https://arxiv.org/pdf/2512.10967)  

**Abstract**: Automatic Speech Recognition (ASR) is increasingly used to document clinical encounters, yet its reliability in multilingual and demographically diverse Indian healthcare contexts remains largely unknown. In this study, we conduct the first systematic audit of ASR performance on real world clinical interview data spanning Kannada, Hindi, and Indian English, comparing leading models including Indic Whisper, Whisper, Sarvam, Google speech to text, Gemma3n, Omnilingual, Vaani, and Gemini. We evaluate transcription accuracy across languages, speakers, and demographic subgroups, with a particular focus on error patterns affecting patients vs. clinicians and gender based or intersectional disparities. Our results reveal substantial variability across models and languages, with some systems performing competitively on Indian English but failing on code mixed or vernacular speech. We also uncover systematic performance gaps tied to speaker role and gender, raising concerns about equitable deployment in clinical settings. By providing a comprehensive multilingual benchmark and fairness analysis, our work highlights the need for culturally and demographically inclusive ASR development for healthcare ecosystem in India. 

---
# From Signal to Turn: Interactional Friction in Modular Speech-to-Speech Pipelines 

**Authors**: Titaya Mairittha, Tanakon Sawanglok, Panuwit Raden, Jirapast Buntub, Thanapat Warunee, Napat Asawachaisuvikrom, Thanaphum Saiwongin  

**Link**: [PDF](https://arxiv.org/pdf/2512.11724)  

**Abstract**: While voice-based AI systems have achieved remarkable generative capabilities, their interactions often feel conversationally broken. This paper examines the interactional friction that emerges in modular Speech-to-Speech Retrieval-Augmented Generation (S2S-RAG) pipelines. By analyzing a representative production system, we move beyond simple latency metrics to identify three recurring patterns of conversational breakdown: (1) Temporal Misalignment, where system delays violate user expectations of conversational rhythm; (2) Expressive Flattening, where the loss of paralinguistic cues leads to literal, inappropriate responses; and (3) Repair Rigidity, where architectural gating prevents users from correcting errors in real-time. Through system-level analysis, we demonstrate that these friction points should not be understood as defects or failures, but as structural consequences of a modular design that prioritizes control over fluidity. We conclude that building natural spoken AI is an infrastructure design challenge, requiring a shift from optimizing isolated components to carefully choreographing the seams between them. 

---
# DentalGPT: Incentivizing Multimodal Complex Reasoning in Dentistry 

**Authors**: Zhenyang Cai, Jiaming Zhang, Junjie Zhao, Ziyi Zeng, Yanchao Li, Jingyi Liang, Junying Chen, Yunjin Yang, Jiajun You, Shuzhi Deng, Tongfei Wang, Wanting Chen, Chunxiu Hao, Ruiqi Xie, Zhenwei Wen, Xiangyi Feng, Zou Ting, Jin Zou Lin, Jianquan Li, Guangjun Yu, Liangyi Chen, Junwen Wang, Shan Jiang, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.11558)  

**Abstract**: Reliable interpretation of multimodal data in dentistry is essential for automated oral healthcare, yet current multimodal large language models (MLLMs) struggle to capture fine-grained dental visual details and lack sufficient reasoning ability for precise diagnosis. To address these limitations, we present DentalGPT, a specialized dental MLLM developed through high-quality domain knowledge injection and reinforcement learning. Specifically, the largest annotated multimodal dataset for dentistry to date was constructed by aggregating over 120k dental images paired with detailed descriptions that highlight diagnostically relevant visual features, making it the multimodal dataset with the most extensive collection of dental images to date. Training on this dataset significantly enhances the MLLM's visual understanding of dental conditions, while the subsequent reinforcement learning stage further strengthens its capability for multimodal complex reasoning. Comprehensive evaluations on intraoral and panoramic benchmarks, along with dental subsets of medical VQA benchmarks, show that DentalGPT achieves superior performance in disease classification and dental VQA tasks, outperforming many state-of-the-art MLLMs despite having only 7B parameters. These results demonstrate that high-quality dental data combined with staged adaptation provides an effective pathway for building capable and domain-specialized dental MLLMs. 

---
# HFS: Holistic Query-Aware Frame Selection for Efficient Video Reasoning 

**Authors**: Yiqing Yang, Kin-Man Lam  

**Link**: [PDF](https://arxiv.org/pdf/2512.11534)  

**Abstract**: Key frame selection in video understanding presents significant challenges. Traditional top-K selection methods, which score frames independently, often fail to optimize the selection as a whole. This independent scoring frequently results in selecting frames that are temporally clustered and visually redundant. Additionally, training lightweight selectors using pseudo labels generated offline by Multimodal Large Language Models (MLLMs) prevents the supervisory signal from dynamically adapting to task objectives. To address these limitations, we propose an end-to-end trainable, task-adaptive framework for frame selection. A Chain-of-Thought approach guides a Small Language Model (SLM) to generate task-specific implicit query vectors, which are combined with multimodal features to enable dynamic frame scoring. We further define a continuous set-level objective function that incorporates relevance, coverage, and redundancy, enabling differentiable optimization via Gumbel-Softmax to select optimal frame combinations at the set level. Finally, student-teacher mutual learning is employed, where the student selector (SLM) and teacher reasoner (MLLM) are trained to align their frame importance distributions via KL divergence. Combined with cross-entropy loss, this enables end-to-end optimization, eliminating reliance on static pseudo labels. Experiments across various benchmarks, including Video-MME, LongVideoBench, MLVU, and NExT-QA, demonstrate that our method significantly outperforms existing approaches. 

---
# Rethinking Expert Trajectory Utilization in LLM Post-training 

**Authors**: Bowen Ding, Yuhan Chen, Jiayang Lv, Jiyao Yuan, Qi Zhu, Shuangshuang Tian, Dantong Zhu, Futing Wang, Heyuan Deng, Fei Mi, Lifeng Shang, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.11470)  

**Abstract**: While effective post-training integrates Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), the optimal mechanism for utilizing expert trajectories remains unresolved. We propose the Plasticity-Ceiling Framework to theoretically ground this landscape, decomposing performance into foundational SFT performance and the subsequent RL plasticity. Through extensive benchmarking, we establish the Sequential SFT-then-RL pipeline as the superior standard, overcoming the stability deficits of synchronized approaches. Furthermore, we derive precise scaling guidelines: (1) Transitioning to RL at the SFT Stable or Mild Overfitting Sub-phase maximizes the final ceiling by securing foundational SFT performance without compromising RL plasticity; (2) Refuting ``Less is More'' in the context of SFT-then-RL scaling, we demonstrate that Data Scale determines the primary post-training potential, while Trajectory Difficulty acts as a performance multiplier; and (3) Identifying that the Minimum SFT Validation Loss serves as a robust indicator for selecting the expert trajectories that maximize the final performance ceiling. Our findings provide actionable guidelines for maximizing the value extracted from expert trajectories. 

---
# Task-Specific Sparse Feature Masks for Molecular Toxicity Prediction with Chemical Language Models 

**Authors**: Kwun Sy Lee, Jiawei Chen, Fuk Sheng Ford Chung, Tianyu Zhao, Zhenyuan Chen, Debby D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.11412)  

**Abstract**: Reliable in silico molecular toxicity prediction is a cornerstone of modern drug discovery, offering a scalable alternative to experimental screening. However, the black-box nature of state-of-the-art models remains a significant barrier to adoption, as high-stakes safety decisions demand verifiable structural insights alongside predictive performance. To address this, we propose a novel multi-task learning (MTL) framework designed to jointly enhance accuracy and interpretability. Our architecture integrates a shared chemical language model with task-specific attention modules. By imposing an L1 sparsity penalty on these modules, the framework is constrained to focus on a minimal set of salient molecular fragments for each distinct toxicity endpoint. The resulting framework is trained end-to-end and is readily adaptable to various transformer-based backbones. Evaluated on the ClinTox, SIDER, and Tox21 benchmark datasets, our approach consistently outperforms both single-task and standard MTL baselines. Crucially, the sparse attention weights provide chemically intuitive visualizations that reveal the specific fragments influencing predictions, thereby enhancing insight into the model's decision-making process. 

---
# Adaptive Soft Rolling KV Freeze with Entropy-Guided Recovery: Sublinear Memory Growth for Efficient LLM Inference 

**Authors**: Adilet Metinov, Gulida M. Kudakeeva, Bolotbek uulu Nursultan, Gulnara D. Kabaeva  

**Link**: [PDF](https://arxiv.org/pdf/2512.11221)  

**Abstract**: We present Adaptive Soft Rolling KV Freeze with Entropy-Guided Recovery (ASR-KF-EGR), a training-free inference-time framework for efficient large language model generation. Our method introduces a reversible soft-freeze mechanism that temporarily suspends key-value (KV) updates for low-importance tokens identified within a sliding attention window. Unlike eviction-based approaches that permanently discard context, ASR-KF-EGR preserves all tokens in off-GPU storage and restores them on demand. We extend the framework with sublinear freeze scheduling, where freeze duration grows sublinearly with repeated low-importance detections, preventing over-aggressive compression. Preliminary experiments on LLaMA-3 8B demonstrate 55-67% reduction in active KV cache size while maintaining generation quality and passing needle-in-haystack retrieval tests. The method is architecture-agnostic, requires no fine-tuning, and provides a practical solution for memory-constrained deployment of long-context LLMs. 

---
# FutureWeaver: Planning Test-Time Compute for Multi-Agent Systems with Modularized Collaboration 

**Authors**: Dongwon Jung, Peng Shi, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.11213)  

**Abstract**: Scaling test-time computation improves large language model performance without additional training. Recent work demonstrates that techniques such as repeated sampling, self-verification, and self-reflection can significantly enhance task success by allocating more inference-time compute. However, applying these techniques across multiple agents in a multi-agent system is difficult: there does not exist principled mechanisms to allocate compute to foster collaboration among agents, to extend test-time scaling to collaborative interactions, or to distribute compute across agents under explicit budget constraints. To address this gap, we propose FutureWeaver, a framework for planning and optimizing test-time compute allocation in multi-agent systems under fixed budgets. FutureWeaver introduces modularized collaboration, formalized as callable functions that encapsulate reusable multi-agent workflows. These modules are automatically derived through self-play reflection by abstracting recurring interaction patterns from past trajectories. Building on these modules, FutureWeaver employs a dual-level planning architecture that optimizes compute allocation by reasoning over the current task state while also speculating on future steps. Experiments on complex agent benchmarks demonstrate that FutureWeaver consistently outperforms baselines across diverse budget settings, validating its effectiveness for multi-agent collaboration in inference-time optimization. 

---
# SCOUT: A Defense Against Data Poisoning Attacks in Fine-Tuned Language Models 

**Authors**: Mohamed Afane, Abhishek Satyam, Ke Chen, Tao Li, Junaid Farooq, Juntao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.10998)  

**Abstract**: Backdoor attacks create significant security threats to language models by embedding hidden triggers that manipulate model behavior during inference, presenting critical risks for AI systems deployed in healthcare and other sensitive domains. While existing defenses effectively counter obvious threats such as out-of-context trigger words and safety alignment violations, they fail against sophisticated attacks using contextually-appropriate triggers that blend seamlessly into natural language. This paper introduces three novel contextually-aware attack scenarios that exploit domain-specific knowledge and semantic plausibility: the ViralApp attack targeting social media addiction classification, the Fever attack manipulating medical diagnosis toward hypertension, and the Referral attack steering clinical recommendations. These attacks represent realistic threats where malicious actors exploit domain-specific vocabulary while maintaining semantic coherence, demonstrating how adversaries can weaponize contextual appropriateness to evade conventional detection methods. To counter both traditional and these sophisticated attacks, we present \textbf{SCOUT (Saliency-based Classification Of Untrusted Tokens)}, a novel defense framework that identifies backdoor triggers through token-level saliency analysis rather than traditional context-based detection methods. SCOUT constructs a saliency map by measuring how the removal of individual tokens affects the model's output logits for the target label, enabling detection of both conspicuous and subtle manipulation attempts. We evaluate SCOUT on established benchmark datasets (SST-2, IMDB, AG News) against conventional attacks (BadNet, AddSent, SynBkd, StyleBkd) and our novel attacks, demonstrating that SCOUT successfully detects these sophisticated threats while preserving accuracy on clean inputs. 

---
# TV2TV: A Unified Framework for Interleaved Language and Video Generation 

**Authors**: Xiaochuang Han, Youssef Emad, Melissa Hall, John Nguyen, Karthik Padthe, Liam Robbins, Amir Bar, Delong Chen, Michal Drozdzal, Maha Elbayad, Yushi Hu, Shang-Wen Li, Sreya Dutta Roy, Jakob Verbeek, XuDong Wang, Marjan Ghazvininejad, Luke Zettlemoyer, Emily Dinan  

**Link**: [PDF](https://arxiv.org/pdf/2512.05103)  

**Abstract**: Video generation models are rapidly advancing, but can still struggle with complex video outputs that require significant semantic branching or repeated high-level reasoning about what should happen next. In this paper, we introduce a new class of omni video-text models that integrate ideas from recent LM reasoning advances to address this challenge. More specifically, we present TV2TV, a unified generative modeling framework which decomposes video generation into an interleaved text and video generation process. TV2TV jointly learns language modeling (next-token prediction) and video flow matching (next-frame prediction) using a Mixture-of-Transformers (MoT) architecture. At inference time, TV2TV decides when to alternate between generating text and video frames, allowing the model to "think in words" about subsequent content before ``acting in pixels'' to produce frames. This design offloads much of the responsibility for deciding what should happen next to the language modeling tower, enabling improved visual quality and prompt alignment of generated videos. It also enables fine-grained controllability, allowing users to modify the video generation trajectory through text interventions at any point in the process. In controlled experiments on video game data, TV2TV demonstrates substantial improvements in both visual quality and controllability. TV2TV also scales to natural videos, as we show by augmenting sports videos with interleaved natural language action descriptions using vision-language models (VLMs). Training TV2TV on this corpus yields strong visual quality and prompt alignment, showcasing the model's ability to reason about and generate complex real-world action sequences. Together, these results highlight TV2TV as a promising step toward video generation with open-ended textual reasoning and control. 

---
