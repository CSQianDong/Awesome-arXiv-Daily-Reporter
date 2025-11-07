# Logit-Entropy Adaptive Stopping Heuristic for Efficient Chain-of-Thought Reasoning 

**Authors**: Mohammad Atif Quamar, Mohammad Areeb  

**Link**: [PDF](https://arxiv.org/pdf/2511.04654)  

**Abstract**: Chain-of-Thought (CoT) prompting is a key technique for enabling complex reasoning in large language models. However, generating full, fixed-length rationales is computationally wasteful, inflating both token usage and latency. We introduce LEASH: Logit-Entropy Adaptive Stopping Heuristic, a training-free decoding algorithm that adaptively halts rationale generation. LEASH monitors two intrinsic signals: the slope of token-level entropy and the improvement in the top-logit margin. It terminates the generation once both signals plateau, indicating the model has reached a stable reasoning state. Across four instruction-tuned models on the GSM8K and AQuA-RAT benchmarks, LEASH reduces average token generation by 30--35% and latency by 27%, while incurring a 10 p.p. accuracy drop relative to CoT. LEASH is model-agnostic and requires no additional training or supervision, offering a simple and efficient alternative to CoT decoding. 

---
# When retrieval outperforms generation: Dense evidence retrieval for scalable fake news detection 

**Authors**: Alamgir Munir Qazi, John P. McCrae, Jamal Abdul Nasir  

**Link**: [PDF](https://arxiv.org/pdf/2511.04643)  

**Abstract**: The proliferation of misinformation necessitates robust yet computationally efficient fact verification systems. While current state-of-the-art approaches leverage Large Language Models (LLMs) for generating explanatory rationales, these methods face significant computational barriers and hallucination risks in real-world deployments. We present DeReC (Dense Retrieval Classification), a lightweight framework that demonstrates how general-purpose text embeddings can effectively replace autoregressive LLM-based approaches in fact verification tasks. By combining dense retrieval with specialized classification, our system achieves better accuracy while being significantly more efficient. DeReC outperforms explanation-generating LLMs in efficiency, reducing runtime by 95% on RAWFC (23 minutes 36 seconds compared to 454 minutes 12 seconds) and by 92% on LIAR-RAW (134 minutes 14 seconds compared to 1692 minutes 23 seconds), showcasing its effectiveness across varying dataset sizes. On the RAWFC dataset, DeReC achieves an F1 score of 65.58%, surpassing the state-of-the-art method L-Defense (61.20%). Our results demonstrate that carefully engineered retrieval-based systems can match or exceed LLM performance in specialized tasks while being significantly more practical for real-world deployment. 

---
# BanglaMedQA and BanglaMMedBench: Evaluating Retrieval-Augmented Generation Strategies for Bangla Biomedical Question Answering 

**Authors**: Sadia Sultana, Saiyma Sittul Muna, Mosammat Zannatul Samarukh, Ajwad Abrar, Tareque Mohmud Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2511.04560)  

**Abstract**: Developing accurate biomedical Question Answering (QA) systems in low-resource languages remains a major challenge, limiting equitable access to reliable medical knowledge. This paper introduces BanglaMedQA and BanglaMMedBench, the first large-scale Bangla biomedical Multiple Choice Question (MCQ) datasets designed to evaluate reasoning and retrieval in medical artificial intelligence (AI). The study applies and benchmarks several Retrieval-Augmented Generation (RAG) strategies, including Traditional, Zero-Shot Fallback, Agentic, Iterative Feedback, and Aggregate RAG, combining textbook-based and web retrieval with generative reasoning to improve factual accuracy. A key novelty lies in integrating a Bangla medical textbook corpus through Optical Character Recognition (OCR) and implementing an Agentic RAG pipeline that dynamically selects between retrieval and reasoning strategies. Experimental results show that the Agentic RAG achieved the highest accuracy 89.54% with openai/gpt-oss-120b, outperforming other configurations and demonstrating superior rationale quality. These findings highlight the potential of RAG-based methods to enhance the reliability and accessibility of Bangla medical QA, establishing a foundation for future research in multilingual medical artificial intelligence. 

---
# From Model to Breach: Towards Actionable LLM-Generated Vulnerabilities Reporting 

**Authors**: Cyril Vallez, Alexander Sternfeld, Andrei Kucharavy, Ljiljana Dolamic  

**Link**: [PDF](https://arxiv.org/pdf/2511.04538)  

**Abstract**: As the role of Large Language Models (LLM)-based coding assistants in software development becomes more critical, so does the role of the bugs they generate in the overall cybersecurity landscape. While a number of LLM code security benchmarks have been proposed alongside approaches to improve the security of generated code, it remains unclear to what extent they have impacted widely used coding LLMs. Here, we show that even the latest open-weight models are vulnerable in the earliest reported vulnerability scenarios in a realistic use setting, suggesting that the safety-functionality trade-off has until now prevented effective patching of vulnerabilities. To help address this issue, we introduce a new severity metric that reflects the risk posed by an LLM-generated vulnerability, accounting for vulnerability severity, generation chance, and the formulation of the prompt that induces vulnerable code generation - Prompt Exposure (PE). To encourage the mitigation of the most serious and prevalent vulnerabilities, we use PE to define the Model Exposure (ME) score, which indicates the severity and prevalence of vulnerabilities a model generates. 

---
# IntelliProof: An Argumentation Network-based Conversational Helper for Organized Reflection 

**Authors**: Kaveh Eskandari Miandoab, Katharine Kowalyshyn, Kabir Pamnani, Anesu Gavhera, Vasanth Sarathy, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2511.04528)  

**Abstract**: We present IntelliProof, an interactive system for analyzing argumentative essays through LLMs. IntelliProof structures an essay as an argumentation graph, where claims are represented as nodes, supporting evidence is attached as node properties, and edges encode supporting or attacking relations. Unlike existing automated essay scoring systems, IntelliProof emphasizes the user experience: each relation is initially classified and scored by an LLM, then visualized for enhanced understanding. The system provides justifications for classifications and produces quantitative measures for essay coherence. It enables rapid exploration of argumentative quality while retaining human oversight. In addition, IntelliProof provides a set of tools for a better understanding of an argumentative essay and its corresponding graph in natural language, bridging the gap between the structural semantics of argumentative essays and the user's understanding of a given text. A live demo and the system are available here to try: \textbf{this https URL} 

---
# Are language models aware of the road not taken? Token-level uncertainty and hidden state dynamics 

**Authors**: Amir Zur, Atticus Geiger, Ekdeep Singh Lubana, Eric Bigelow  

**Link**: [PDF](https://arxiv.org/pdf/2511.04527)  

**Abstract**: When a language model generates text, the selection of individual tokens might lead it down very different reasoning paths, making uncertainty difficult to quantify. In this work, we consider whether reasoning language models represent the alternate paths that they could take during generation. To test this hypothesis, we use hidden activations to control and predict a language model's uncertainty during chain-of-thought reasoning. In our experiments, we find a clear correlation between how uncertain a model is at different tokens, and how easily the model can be steered by controlling its activations. This suggests that activation interventions are most effective when there are alternate paths available to the model -- in other words, when it has not yet committed to a particular final answer. We also find that hidden activations can predict a model's future outcome distribution, demonstrating that models implicitly represent the space of possible paths. 

---
# Modeling Clinical Uncertainty in Radiology Reports: from Explicit Uncertainty Markers to Implicit Reasoning Pathways 

**Authors**: Paloma Rabaey, Jong Hak Moon, Jung-Oh Lee, Min Gwan Kim, Hangyul Yoon, Thomas Demeester, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04506)  

**Abstract**: Radiology reports are invaluable for clinical decision-making and hold great potential for automated analysis when structured into machine-readable formats. These reports often contain uncertainty, which we categorize into two distinct types: (i) Explicit uncertainty reflects doubt about the presence or absence of findings, conveyed through hedging phrases. These vary in meaning depending on the context, making rule-based systems insufficient to quantify the level of uncertainty for specific findings; (ii) Implicit uncertainty arises when radiologists omit parts of their reasoning, recording only key findings or diagnoses. Here, it is often unclear whether omitted findings are truly absent or simply unmentioned for brevity. We address these challenges with a two-part framework. We quantify explicit uncertainty by creating an expert-validated, LLM-based reference ranking of common hedging phrases, and mapping each finding to a probability value based on this reference. In addition, we model implicit uncertainty through an expansion framework that systematically adds characteristic sub-findings derived from expert-defined diagnostic pathways for 14 common diagnoses. Using these methods, we release Lunguage++, an expanded, uncertainty-aware version of the Lunguage benchmark of fine-grained structured radiology reports. This enriched resource enables uncertainty-aware image classification, faithful diagnostic reasoning, and new investigations into the clinical impact of diagnostic uncertainty. 

---
# RAGalyst: Automated Human-Aligned Agentic Evaluation for Domain-Specific RAG 

**Authors**: Joshua Gao, Quoc Huy Pham, Subin Varghese, Silwal Saurav, Vedhus Hoskere  

**Link**: [PDF](https://arxiv.org/pdf/2511.04502)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a critical technique for grounding Large Language Models (LLMs) in factual evidence, yet evaluating RAG systems in specialized, safety-critical domains remains a significant challenge. Existing evaluation frameworks often rely on heuristic-based metrics that fail to capture domain-specific nuances and other works utilize LLM-as-a-Judge approaches that lack validated alignment with human judgment. This paper introduces RAGalyst, an automated, human-aligned agentic framework designed for the rigorous evaluation of domain-specific RAG systems. RAGalyst features an agentic pipeline that generates high-quality, synthetic question-answering (QA) datasets from source documents, incorporating an agentic filtering step to ensure data fidelity. The framework refines two key LLM-as-a-Judge metrics-Answer Correctness and Answerability-using prompt optimization to achieve a strong correlation with human annotations. Applying this framework to evaluate various RAG components across three distinct domains (military operations, cybersecurity, and bridge engineering), we find that performance is highly context-dependent. No single embedding model, LLM, or hyperparameter configuration proves universally optimal. Additionally, we provide an analysis on the most common low Answer Correctness reasons in RAG. These findings highlight the necessity of a systematic evaluation framework like RAGalyst, which empowers practitioners to uncover domain-specific trade-offs and make informed design choices for building reliable and effective RAG systems. RAGalyst is available on our Github. 

---
# Decoding Emergent Big Five Traits in Large Language Models: Temperature-Dependent Expression and Architectural Clustering 

**Authors**: Christos-Nikolaos Zacharopoulos, Revekka Kyriakoglou  

**Link**: [PDF](https://arxiv.org/pdf/2511.04499)  

**Abstract**: As Large Language Models (LLMs) become integral to human-centered applications, understanding their personality-like behaviors is increasingly important for responsible development and deployment. This paper systematically evaluates six LLMs, applying the Big Five Inventory-2 (BFI-2) framework, to assess trait expressions under varying sampling temperatures. We find significant differences across four of the five personality dimensions, with Neuroticism and Extraversion susceptible to temperature adjustments. Further, hierarchical clustering reveals distinct model clusters, suggesting that architectural features may predispose certain models toward stable trait profiles. Taken together, these results offer new insights into the emergence of personality-like patterns in LLMs and provide a new perspective on model tuning, selection, and the ethical governance of AI systems. We share the data and code for this analysis here: this https URL 

---
# OUNLP at TSAR 2025 Shared Task: Multi-Round Text Simplifier via Code Generation 

**Authors**: Cuong Huynh, Jie Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.04495)  

**Abstract**: This paper describes the OUNLP system submitted to the TSAR-2025 Shared Task (Alva-Manchego et al., 2025), designed for readability-controlled text simplification using LLM-prompting-based generation. Based on the analysis of prompt-based text simplification methods, we discovered an interesting finding that text simplification performance is highly related to the gap between the source CEFR (Arase et al., 2022) level and the target CEFR level. Inspired by this finding, we propose two multi-round simplification methods and generate them via GPT-4o: rule-based simplification (MRS-Rule) and jointly rule-based LLM simplification (MRS-Joint). Our submitted systems ranked 7 out of 20 teams. Later improvements with MRS-Joint show that taking the LLM simplified candidates as the starting point could further boost the multi-round simplification performance. 

---
# RUST-BENCH: Benchmarking LLM Reasoning on Unstructured Text within Structured Tables 

**Authors**: Nikhil Abhyankar, Purvi Chaurasia, Sanchit Kabra, Ananya Srivastava, Vivek Gupta, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2511.04491)  

**Abstract**: Existing tabular reasoning benchmarks mostly test models on small, uniform tables, underrepresenting the complexity of real-world data and giving an incomplete view of Large Language Models' (LLMs) reasoning abilities. Real tables are long, heterogeneous, and domain-specific, mixing structured fields with free text and requiring multi-hop reasoning across thousands of tokens. To address this gap, we introduce RUST-BENCH, a benchmark of 7966 questions from 2031 real-world tables spanning two domains: i) RB-Science (NSF grant records) and ii) RB-Sports (NBA statistics). Unlike prior work, RUST-BENCH evaluates LLMs jointly across scale, heterogeneity, domain specificity, and reasoning complexity. Experiments with open-source and proprietary models show that LLMs struggle with heterogeneous schemas and complex multi-hop inference, revealing persistent weaknesses in current architectures and prompting strategies. RUST-BENCH establishes a challenging new testbed for advancing tabular reasoning research. 

---
# ThaiOCRBench: A Task-Diverse Benchmark for Vision-Language Understanding in Thai 

**Authors**: Surapon Nonesung, Teetouch Jaknamon, Sirinya Chaiophat, Natapong Nitarach, Chanakan Wittayasakpan, Warit Sirichotedumrong, Adisai Na-Thalang, Kunat Pipatanakul  

**Link**: [PDF](https://arxiv.org/pdf/2511.04479)  

**Abstract**: We present ThaiOCRBench, the first comprehensive benchmark for evaluating vision-language models (VLMs) on Thai text-rich visual understanding tasks. Despite recent progress in multimodal modeling, existing benchmarks predominantly focus on high-resource languages, leaving Thai underrepresented, especially in tasks requiring document structure understanding. ThaiOCRBench addresses this gap by offering a diverse, human-annotated dataset comprising 2,808 samples across 13 task categories. We evaluate a wide range of state-of-the-art VLMs in a zero-shot setting, spanning both proprietary and open-source systems. Results show a significant performance gap, with proprietary models (e.g., Gemini 2.5 Pro) outperforming open-source counterparts. Notably, fine-grained text recognition and handwritten content extraction exhibit the steepest performance drops among open-source models. Through detailed error analysis, we identify key challenges such as language bias, structural mismatch, and hallucinated content. ThaiOCRBench provides a standardized framework for assessing VLMs in low-resource, script-complex settings, and provides actionable insights for improving Thai-language document understanding. 

---
# Probabilistic Textual Time Series Depression Detection 

**Authors**: Fabian Schmidt, Seyedehmoniba Ravan, Vladimir Vlassov  

**Link**: [PDF](https://arxiv.org/pdf/2511.04476)  

**Abstract**: Accurate and interpretable predictions of depression severity are essential for clinical decision support, yet existing models often lack uncertainty estimates and temporal modeling. We propose PTTSD, a Probabilistic Textual Time Series Depression Detection framework that predicts PHQ-8 scores from utterance-level clinical interviews while modeling uncertainty over time. PTTSD includes sequence-to-sequence and sequence-to-one variants, both combining bidirectional LSTMs, self-attention, and residual connections with Gaussian or Student-t output heads trained via negative log-likelihood. Evaluated on E-DAIC and DAIC-WOZ, PTTSD achieves state-of-the-art performance among text-only systems (e.g., MAE = 3.85 on E-DAIC, 3.55 on DAIC) and produces well-calibrated prediction intervals. Ablations confirm the value of attention and probabilistic modeling, while comparisons with MentalBERT establish generality. A three-part calibration analysis and qualitative case studies further highlight the interpretability and clinical relevance of uncertainty-aware forecasting. 

---
# If I Could Turn Back Time: Temporal Reframing as a Historical Reasoning Task for LLMs 

**Authors**: Lars Bungum, Charles Yijia Huang, Abeer Kashar  

**Link**: [PDF](https://arxiv.org/pdf/2511.04432)  

**Abstract**: In this study, we experiment with the ability of LLMs to do temporal reasoning. Using a Norwegian book from 1940 containing trivia questions, we prompt the LLMs to answer the questions as if it were 1940. We also pose the questions in both English and Norwegian. Correct answers are often presented as sentences, and grading is done by means of LLM-as-judge, with sampled checks by a native speaker. Prompting in English consistently gave better results than in Norwegian, an unexpected result. In contrast, using larger LLMs improved results. We tested the DeepSeek-R1, Gemma3, Qwen3, and Llama3.1 model families, and also the largest available LLM especially crafted for Norwegian. 

---
# Dynamic Jointly Batch Selection for Data Efficient Machine Translation Fine-Tuning 

**Authors**: Mohammad Amin Ghanizadeh, Mohammad Javad Dousti  

**Link**: [PDF](https://arxiv.org/pdf/2511.04406)  

**Abstract**: Data quality and its effective selection are fundamental to improving the performance of machine translation models, serving as cornerstones for achieving robust and reliable translation systems. This paper presents a data selection methodology specifically designed for fine-tuning machine translation systems, which leverages the synergy between a learner model and a pre-trained reference model to enhance overall training effectiveness. By defining a learnability score, our approach systematically evaluates the utility of data points for training, ensuring that only the most relevant and impactful examples contribute to the fine-tuning process. Furthermore, our method employs a batch selection strategy which considers interdependencies among data points, optimizing the efficiency of the training process while maintaining a focus on data relevance. Experiments on English to Persian and several other language pairs using an mBART model fine-tuned on the CCMatrix dataset demonstrate that our method can achieve up to a fivefold improvement in data efficiency compared to an iid baseline. Experimental results indicate that our approach improves computational efficiency by 24 when utilizing cached embeddings, as it requires fewer training data points. Additionally, it enhances generalization, resulting in superior translation performance compared to random selection method. 

---
# SSPO: Subsentence-level Policy Optimization 

**Authors**: Kun Yang, Zikang chen, Yanmeng Wang, Zhigen Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.04256)  

**Abstract**: As a significant part of post-training of the Large Language Models (LLMs), Reinforcement Learning from Verifiable Reward (RLVR) has greatly improved LLMs' reasoning skills. However, some RLVR algorithms, such as GRPO (Group Relative Policy Optimization) and GSPO (Group Sequence Policy Optimization), are observed to suffer from unstable policy updates and low usage of sampling data, respectively. The importance ratio of GRPO is calculated at the token level, which focuses more on optimizing a single token. This will be easily affected by outliers, leading to model training collapse. GSPO proposed the calculation of the response level importance ratio, which solves the problem of high variance and training noise accumulation in the calculation of the GRPO importance ratio. However, since all the response tokens share a common importance ratio, extreme values can easily raise or lower the overall mean, leading to the entire response being mistakenly discarded, resulting in a decrease in the utilization of sampled data. This paper introduces SSPO, which applies sentence-level importance ratio, taking the balance between GRPO and GSPO. SSPO not only avoids training collapse and high variance, but also prevents the whole response tokens from being abandoned by the clipping mechanism. Furthermore, we apply sentence entropy to PPO-CLIP to steadily adjust the clipping bounds, encouraging high-entropy tokens to explore and narrow the clipping range of low-entropy tokens. In particular, SSPO achieves an average score of 46.57 across five datasets, surpassing GRPO (43.01) and GSPO (44.42), and wins state-of-the-art performance on three datasets. These results highlight SSPO's effectiveness in leveraging generated data by taking the essence of GSPO but rejecting its shortcomings. 

---
# Efficient Topic Extraction via Graph-Based Labeling: A Lightweight Alternative to Deep Models 

**Authors**: Salma Mekaooui, Hiba Sofyan, Imane Amaaz, Imane Benchrif, Arsalane Zarghili, Ilham Chaker, Nikola S. Nikolov  

**Link**: [PDF](https://arxiv.org/pdf/2511.04248)  

**Abstract**: Extracting topics from text has become an essential task, especially with the rapid growth of unstructured textual data. Most existing works rely on highly computational methods to address this challenge. In this paper, we argue that probabilistic and statistical approaches, such as topic modeling (TM), can offer effective alternatives that require fewer computational resources. TM is a statistical method that automatically discovers topics in large collections of unlabeled text; however, it produces topics as distributions of representative words, which often lack clear interpretability. Our objective is to perform topic labeling by assigning meaningful labels to these sets of words. To achieve this without relying on computationally expensive models, we propose a graph-based approach that not only enriches topic words with semantically related terms but also explores the relationships among them. By analyzing these connections within the graph, we derive suitable labels that accurately capture each topic's meaning. We present a comparative study between our proposed method and several benchmarks, including ChatGPT-3.5, across two different datasets. Our method achieved consistently better results than traditional benchmarks in terms of BERTScore and cosine similarity and produced results comparable to ChatGPT-3.5, while remaining computationally efficient. Finally, we discuss future directions for topic labeling and highlight potential research avenues for enhancing interpretability and automation. 

---
# Reusing Pre-Training Data at Test Time is a Compute Multiplier 

**Authors**: Alex Fang, Thomas Voice, Ruoming Pang, Ludwig Schmidt, Tom Gunter  

**Link**: [PDF](https://arxiv.org/pdf/2511.04234)  

**Abstract**: Large language models learn from their vast pre-training corpora, gaining the ability to solve an ever increasing variety of tasks; yet although researchers work to improve these datasets, there is little effort to understand how efficient the pre-training apparatus is at extracting ideas and knowledge from the data. In this work, we use retrieval augmented generation along with test-time compute as a way to quantify how much dataset value was left behind by the process of pre-training, and how this changes across scale. We demonstrate that pre-training then retrieving from standard and largely open-sourced datasets results in significant accuracy gains in MMLU, Math-500, and SimpleQA, which persist through decontamination. For MMLU we observe that retrieval acts as a ~5x compute multiplier versus pre-training alone. We show that these results can be further improved by leveraging additional compute at test time to parse the retrieved context, demonstrating a 10 percentage point improvement on MMLU for the public LLaMA 3.1 8B model. Overall, our results suggest that today's pre-training methods do not make full use of the information in existing pre-training datasets, leaving significant room for progress. 

---
# REMIND: Input Loss Landscapes Reveal Residual Memorization in Post-Unlearning LLMs 

**Authors**: Liran Cohen, Yaniv Nemcovesky, Avi Mendelson  

**Link**: [PDF](https://arxiv.org/pdf/2511.04228)  

**Abstract**: Machine unlearning aims to remove the influence of specific training data from a model without requiring full retraining. This capability is crucial for ensuring privacy, safety, and regulatory compliance. Therefore, verifying whether a model has truly forgotten target data is essential for maintaining reliability and trustworthiness. However, existing evaluation methods often assess forgetting at the level of individual inputs. This approach may overlook residual influence present in semantically similar examples. Such influence can compromise privacy and lead to indirect information leakage. We propose REMIND (Residual Memorization In Neighborhood Dynamics), a novel evaluation method aiming to detect the subtle remaining influence of unlearned data and classify whether the data has been effectively forgotten. REMIND analyzes the model's loss over small input variations and reveals patterns unnoticed by single-point evaluations. We show that unlearned data yield flatter, less steep loss landscapes, while retained or unrelated data exhibit sharper, more volatile patterns. REMIND requires only query-based access, outperforms existing methods under similar constraints, and demonstrates robustness across different models, datasets, and paraphrased inputs, making it practical for real-world deployment. By providing a more sensitive and interpretable measure of unlearning effectiveness, REMIND provides a reliable framework to assess unlearning in language models. As a result, REMIND offers a novel perspective on memorization and unlearning. 

---
# LLM-as-a-Judge is Bad, Based on AI Attempting the Exam Qualifying for the Member of the Polish National Board of Appeal 

**Authors**: Michał Karp, Anna Kubaszewska, Magdalena Król, Robert Król, Aleksander Smywiński-Pohl, Mateusz Szymański, Witold Wydmański  

**Link**: [PDF](https://arxiv.org/pdf/2511.04205)  

**Abstract**: This study provides an empirical assessment of whether current large language models (LLMs) can pass the official qualifying examination for membership in Poland's National Appeal Chamber (Krajowa Izba Odwoławcza). The authors examine two related ideas: using LLM as actual exam candidates and applying the 'LLM-as-a-judge' approach, in which model-generated answers are automatically evaluated by other models. The paper describes the structure of the exam, which includes a multiple-choice knowledge test on public procurement law and a written judgment, and presents the hybrid information recovery and extraction pipeline built to support the models. Several LLMs (including GPT-4.1, Claude 4 Sonnet and Bielik-11B-v2.6) were tested in closed-book and various Retrieval-Augmented Generation settings. The results show that although the models achieved satisfactory scores in the knowledge test, none met the passing threshold in the practical written part, and the evaluations of the 'LLM-as-a-judge' often diverged from the judgments of the official examining committee. The authors highlight key limitations: susceptibility to hallucinations, incorrect citation of legal provisions, weaknesses in logical argumentation, and the need for close collaboration between legal experts and technical teams. The findings indicate that, despite rapid technological progress, current LLMs cannot yet replace human judges or independent examiners in Polish public procurement adjudication. 

---
# Computational Turing Test Reveals Systematic Differences Between Human and AI Language 

**Authors**: Nicolò Pagan, Petter Törnberg, Christopher A. Bail, Anikó Hannák, Christopher Barrie  

**Link**: [PDF](https://arxiv.org/pdf/2511.04195)  

**Abstract**: Large language models (LLMs) are increasingly used in the social sciences to simulate human behavior, based on the assumption that they can generate realistic, human-like text. Yet this assumption remains largely untested. Existing validation efforts rely heavily on human-judgment-based evaluations -- testing whether humans can distinguish AI from human output -- despite evidence that such judgments are blunt and unreliable. As a result, the field lacks robust tools for assessing the realism of LLM-generated text or for calibrating models to real-world data. This paper makes two contributions. First, we introduce a computational Turing test: a validation framework that integrates aggregate metrics (BERT-based detectability and semantic similarity) with interpretable linguistic features (stylistic markers and topical patterns) to assess how closely LLMs approximate human language within a given dataset. Second, we systematically compare nine open-weight LLMs across five calibration strategies -- including fine-tuning, stylistic prompting, and context retrieval -- benchmarking their ability to reproduce user interactions on X (formerly Twitter), Bluesky, and Reddit. Our findings challenge core assumptions in the literature. Even after calibration, LLM outputs remain clearly distinguishable from human text, particularly in affective tone and emotional expression. Instruction-tuned models underperform their base counterparts, and scaling up model size does not enhance human-likeness. Crucially, we identify a trade-off: optimizing for human-likeness often comes at the cost of semantic fidelity, and vice versa. These results provide a much-needed scalable framework for validation and calibration in LLM simulations -- and offer a cautionary note about their current limitations in capturing human communication. 

---
# Trustworthy LLM-Mediated Communication: Evaluating Information Fidelity in LLM as a Communicator (LAAC) Framework in Multiple Application Domains 

**Authors**: Mohammed Musthafa Rafi, Adarsh Krishnamurthy, Aditya Balu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04184)  

**Abstract**: The proliferation of AI-generated content has created an absurd communication theater where senders use LLMs to inflate simple ideas into verbose content, recipients use LLMs to compress them back into summaries, and as a consequence neither party engage with authentic content. LAAC (LLM as a Communicator) proposes a paradigm shift - positioning LLMs as intelligent communication intermediaries that capture the sender's intent through structured dialogue and facilitate genuine knowledge exchange with recipients. Rather than perpetuating cycles of AI-generated inflation and compression, LAAC enables authentic communication across diverse contexts including academic papers, proposals, professional emails, and cross-platform content generation. However, deploying LLMs as trusted communication intermediaries raises critical questions about information fidelity, consistency, and reliability. This position paper systematically evaluates the trustworthiness requirements for LAAC's deployment across multiple communication domains. We investigate three fundamental dimensions: (1) Information Capture Fidelity - accuracy of intent extraction during sender interviews across different communication types, (2) Reproducibility - consistency of structured knowledge across multiple interaction instances, and (3) Query Response Integrity - reliability of recipient-facing responses without hallucination, source conflation, or fabrication. Through controlled experiments spanning multiple LAAC use cases, we assess these trust dimensions using LAAC's multi-agent architecture. Preliminary findings reveal measurable trust gaps that must be addressed before LAAC can be reliably deployed in high-stakes communication scenarios. 

---
# BAPPA: Benchmarking Agents, Plans, and Pipelines for Automated Text-to-SQL Generation 

**Authors**: Fahim Ahmed, Md Mubtasim Ahasan, Jahir Sadik Monon, Muntasir Wahed, M Ashraful Amin, A K M Mahbubur Rahman, Amin Ahsan Ali  

**Link**: [PDF](https://arxiv.org/pdf/2511.04153)  

**Abstract**: Text-to-SQL systems provide a natural language interface that can enable even laymen to access information stored in databases. However, existing Large Language Models (LLM) struggle with SQL generation from natural instructions due to large schema sizes and complex reasoning. Prior work often focuses on complex, somewhat impractical pipelines using flagship models, while smaller, efficient models remain overlooked. In this work, we explore three multi-agent LLM pipelines, with systematic performance benchmarking across a range of small to large open-source models: (1) Multi-agent discussion pipeline, where agents iteratively critique and refine SQL queries, and a judge synthesizes the final answer; (2) Planner-Coder pipeline, where a thinking model planner generates stepwise SQL generation plans and a coder synthesizes queries; and (3) Coder-Aggregator pipeline, where multiple coders independently generate SQL queries, and a reasoning agent selects the best query. Experiments on the Bird-Bench Mini-Dev set reveal that Multi-Agent discussion can improve small model performance, with up to 10.6% increase in Execution Accuracy for Qwen2.5-7b-Instruct seen after three rounds of discussion. Among the pipelines, the LLM Reasoner-Coder pipeline yields the best results, with DeepSeek-R1-32B and QwQ-32B planners boosting Gemma 3 27B IT accuracy from 52.4% to the highest score of 56.4%. Codes are available at this https URL. 

---
# CantoASR: Prosody-Aware ASR-LALM Collaboration for Low-Resource Cantonese 

**Authors**: Dazhong Chen, Yi-Cheng Lin, Yuchen Huang, Ziwei Gong, Di Jiang, Zeying Xie, Yi R., Fung  

**Link**: [PDF](https://arxiv.org/pdf/2511.04139)  

**Abstract**: Automatic speech recognition (ASR) is critical for language accessibility, yet low-resource Cantonese remains challenging due to limited annotated data, six lexical tones, tone sandhi, and accent variation. Existing ASR models, such as Whisper, often suffer from high word error rates. Large audio-language models (LALMs), in contrast, can leverage broader contextual reasoning but still require explicit tonal and prosodic acoustic cues. We introduce CantoASR, a collaborative ASR-LALM error correction framework that integrates forced alignment for acoustic feature extraction, a LoRA-finetuned Whisper for improved tone discrimination, and an instruction-tuned Qwen-Audio for prosody-aware correction. Evaluations on spontaneous Cantonese data show substantial CER gains over Whisper-Large-V3. These findings suggest that integrating acoustic cues with LALM reasoning provides a scalable strategy for low-resource tonal and dialectal ASR. 

---
# RIDE: Difficulty Evolving Perturbation with Item Response Theory for Mathematical Reasoning 

**Authors**: Xinyuan Li, Murong Xu, Wenbiao Tao, Hanlun Zhu, Yike Zhao, Jipeng Zhang, Yunshi Lan  

**Link**: [PDF](https://arxiv.org/pdf/2511.04120)  

**Abstract**: Large language models (LLMs) achieve high performance on mathematical reasoning, but these results can be inflated by training data leakage or superficial pattern matching rather than genuine reasoning. To this end, an adversarial perturbation-based evaluation is needed to measure true mathematical reasoning ability. Current rule-based perturbation methods often generate ill-posed questions and impede the systematic evaluation of question difficulty and the evolution of benchmarks. To bridge this gap, we propose RIDE, a novel adversarial question-rewriting framework that leverages Item Response Theory (IRT) to rigorously measure question difficulty and to generate intrinsically more challenging, well-posed variations of mathematical problems. We employ 35 LLMs to simulate students and build a difficulty ranker from their responses. This ranker provides a reward signal during reinforcement learning and guides a question-rewriting model to reformulate existing questions across difficulty levels. Applying RIDE to competition-level mathematical benchmarks yields perturbed versions that degrade advanced LLM performance, with experiments showing an average 21.73% drop across 26 models, thereby exposing limited robustness in mathematical reasoning and confirming the validity of our evaluation approach. 

---
# Batch Prompting Suppresses Overthinking Reasoning Under Constraint: How Batch Prompting Suppresses Overthinking in Reasoning Models 

**Authors**: Wenmo Qiu, Saurabh Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2511.04108)  

**Abstract**: Recent work has explored batch prompting as a strategy to amortize inference cost in large language models (LLMs). In this paper, we show that batching offers an additional, underappreciated benefit: it regularizes model behavior during multi-step reasoning for Large Reasoning Models (LRMs). We conduct a comprehensive study across 13 diverse benchmarks and observe that batching improves accuracy while substantially reducing reasoning token usage, often by 3x-5x. Through detailed behavioral analysis, we find that batching suppresses overthinking, reduces hedging language (e.g., repetitive self-corrections), and encourages more decisive answers. Surprisingly, we also observe emergent collective effects in batched inference: models often generalize patterns from earlier examples to solve harder ones in the same batch. These findings position batching not just as a throughput optimization, but as a powerful inference-time regularizer for more efficient and reliable LLM reasoning. 

---
# A Characterization of List Language Identification in the Limit 

**Authors**: Moses Charikar, Chirag Pabbaraju, Ambuj Tewari  

**Link**: [PDF](https://arxiv.org/pdf/2511.04103)  

**Abstract**: We study the problem of language identification in the limit, where given a sequence of examples from a target language, the goal of the learner is to output a sequence of guesses for the target language such that all the guesses beyond some finite time are correct. Classical results of Gold showed that language identification in the limit is impossible for essentially any interesting collection of languages. Later, Angluin gave a precise characterization of language collections for which this task is possible. Motivated by recent positive results for the related problem of language generation, we revisit the classic language identification problem in the setting where the learner is given the additional power of producing a list of $k$ guesses at each time step. The goal is to ensure that beyond some finite time, one of the guesses is correct at each time step.
We give an exact characterization of collections of languages that can be $k$-list identified in the limit, based on a recursive version of Angluin's characterization (for language identification with a list of size $1$). This further leads to a conceptually appealing characterization: A language collection can be $k$-list identified in the limit if and only if the collection can be decomposed into $k$ collections of languages, each of which can be identified in the limit (with a list of size $1$). We also use our characterization to establish rates for list identification in the statistical setting where the input is drawn as an i.i.d. stream from a distribution supported on some language in the collection. Our results show that if a collection is $k$-list identifiable in the limit, then the collection can be $k$-list identified at an exponential rate, and this is best possible. On the other hand, if a collection is not $k$-list identifiable in the limit, then it cannot be $k$-list identified at any rate that goes to zero. 

---
# Improving the Performance of Radiology Report De-identification with Large-Scale Training and Benchmarking Against Cloud Vendor Methods 

**Authors**: Eva Prakash, Maayane Attias, Pierre Chambon, Justin Xu, Steven Truong, Jean-Benoit Delbrouck, Tessa Cook, Curtis Langlotz  

**Link**: [PDF](https://arxiv.org/pdf/2511.04079)  

**Abstract**: Objective: To enhance automated de-identification of radiology reports by scaling transformer-based models through extensive training datasets and benchmarking performance against commercial cloud vendor systems for protected health information (PHI) detection. Materials and Methods: In this retrospective study, we built upon a state-of-the-art, transformer-based, PHI de-identification pipeline by fine-tuning on two large annotated radiology corpora from Stanford University, encompassing chest X-ray, chest CT, abdomen/pelvis CT, and brain MR reports and introducing an additional PHI category (AGE) into the architecture. Model performance was evaluated on test sets from Stanford and the University of Pennsylvania (Penn) for token-level PHI detection. We further assessed (1) the stability of synthetic PHI generation using a "hide-in-plain-sight" method and (2) performance against commercial systems. Precision, recall, and F1 scores were computed across all PHI categories. Results: Our model achieved overall F1 scores of 0.973 on the Penn dataset and 0.996 on the Stanford dataset, outperforming or maintaining the previous state-of-the-art model performance. Synthetic PHI evaluation showed consistent detectability (overall F1: 0.959 [0.958-0.960]) across 50 independently de-identified Penn datasets. Our model outperformed all vendor systems on synthetic Penn reports (overall F1: 0.960 vs. 0.632-0.754). Discussion: Large-scale, multimodal training improved cross-institutional generalization and robustness. Synthetic PHI generation preserved data utility while ensuring privacy. Conclusion: A transformer-based de-identification model trained on diverse radiology datasets outperforms prior academic and commercial systems in PHI detection and establishes a new benchmark for secure clinical text processing. 

---
# The truth is no diaper: Human and AI-generated associations to emotional words 

**Authors**: Špela Vintar, Jan Jona Javoršek  

**Link**: [PDF](https://arxiv.org/pdf/2511.04077)  

**Abstract**: Human word associations are a well-known method of gaining insight into the internal mental lexicon, but the responses spontaneously offered by human participants to word cues are not always predictable as they may be influenced by personal experience, emotions or individual cognitive styles. The ability to form associative links between seemingly unrelated concepts can be the driving mechanisms of creativity. We perform a comparison of the associative behaviour of humans compared to large language models. More specifically, we explore associations to emotionally loaded words and try to determine whether large language models generate associations in a similar way to humans. We find that the overlap between humans and LLMs is moderate, but also that the associations of LLMs tend to amplify the underlying emotional load of the stimulus, and that they tend to be more predictable and less creative than human ones. 

---
# Plan of Knowledge: Retrieval-Augmented Large Language Models for Temporal Knowledge Graph Question Answering 

**Authors**: Xinying Qian, Ying Zhang, Yu Zhao, Baohang Zhou, Xuhui Sui, Xiaojie Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2511.04072)  

**Abstract**: Temporal Knowledge Graph Question Answering (TKGQA) aims to answer time-sensitive questions by leveraging factual information from Temporal Knowledge Graphs (TKGs). While previous studies have employed pre-trained TKG embeddings or graph neural networks to inject temporal knowledge, they fail to fully understand the complex semantic information of time constraints. Recently, Large Language Models (LLMs) have shown remarkable progress, benefiting from their strong semantic understanding and reasoning generalization capabilities. However, their temporal reasoning ability remains limited. LLMs frequently suffer from hallucination and a lack of knowledge. To address these limitations, we propose the Plan of Knowledge framework with a contrastive temporal retriever, which is named PoK. Specifically, the proposed Plan of Knowledge module decomposes a complex temporal question into a sequence of sub-objectives from the pre-defined tools, serving as intermediate guidance for reasoning exploration. In parallel, we construct a Temporal Knowledge Store (TKS) with a contrastive retrieval framework, enabling the model to selectively retrieve semantically and temporally aligned facts from TKGs. By combining structured planning with temporal knowledge retrieval, PoK effectively enhances the interpretability and factual consistency of temporal reasoning. Extensive experiments on four benchmark TKGQA datasets demonstrate that PoK significantly improves the retrieval precision and reasoning accuracy of LLMs, surpassing the performance of the state-of-the-art TKGQA methods by 56.0% at most. 

---
# T-FIX: Text-Based Explanations with Features Interpretable to eXperts 

**Authors**: Shreya Havaldar, Helen Jin, Chaehyeon Kim, Anton Xue, Weiqiu You, Marco Gatti, Bhuvnesh Jain, Helen Qu, Daniel A Hashimoto, Amin Madani, Rajat Deo, Sameed Ahmed M. Khatana, Gary E. Weissman, Lyle Ungar, Eric Wong  

**Link**: [PDF](https://arxiv.org/pdf/2511.04070)  

**Abstract**: As LLMs are deployed in knowledge-intensive settings (e.g., surgery, astronomy, therapy), users expect not just answers, but also meaningful explanations for those answers. In these settings, users are often domain experts (e.g., doctors, astrophysicists, psychologists) who require explanations that reflect expert-level reasoning. However, current evaluation schemes primarily emphasize plausibility or internal faithfulness of the explanation, which fail to capture whether the content of the explanation truly aligns with expert intuition. We formalize expert alignment as a criterion for evaluating explanations with T-FIX, a benchmark spanning seven knowledge-intensive domains. In collaboration with domain experts, we develop novel metrics to measure the alignment of LLM explanations with expert judgment. 

---
# WST: Weakly Supervised Transducer for Automatic Speech Recognition 

**Authors**: Dongji Gao, Chenda Liao, Changliang Liu, Matthew Wiesner, Leibny Paola Garcia, Daniel Povey, Sanjeev Khudanpur, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04035)  

**Abstract**: The Recurrent Neural Network-Transducer (RNN-T) is widely adopted in end-to-end (E2E) automatic speech recognition (ASR) tasks but depends heavily on large-scale, high-quality annotated data, which are often costly and difficult to obtain. To mitigate this reliance, we propose a Weakly Supervised Transducer (WST), which integrates a flexible training graph designed to robustly handle errors in the transcripts without requiring additional confidence estimation or auxiliary pre-trained models. Empirical evaluations on synthetic and industrial datasets reveal that WST effectively maintains performance even with transcription error rates of up to 70%, consistently outperforming existing Connectionist Temporal Classification (CTC)-based weakly supervised approaches, such as Bypass Temporal Classification (BTC) and Omni-Temporal Classification (OTC). These results demonstrate the practical utility and robustness of WST in realistic ASR settings. The implementation will be publicly available. 

---
# Abductive Inference in Retrieval-Augmented Language Models: Generating and Validating Missing Premises 

**Authors**: Shiyin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.04020)  

**Abstract**: Large Language Models (LLMs) enhanced with retrieval -- commonly referred to as Retrieval-Augmented Generation (RAG) -- have demonstrated strong performance in knowledge-intensive tasks. However, RAG pipelines often fail when retrieved evidence is incomplete, leaving gaps in the reasoning process. In such cases, \emph{abductive inference} -- the process of generating plausible missing premises to explain observations -- offers a principled approach to bridge these gaps. In this paper, we propose a framework that integrates abductive inference into retrieval-augmented LLMs. Our method detects insufficient evidence, generates candidate missing premises, and validates them through consistency and plausibility checks. Experimental results on abductive reasoning and multi-hop QA benchmarks show that our approach improves both answer accuracy and reasoning faithfulness. This work highlights abductive inference as a promising direction for enhancing the robustness and explainability of RAG systems. 

---
# Direct Semantic Communication Between Large Language Models via Vector Translation 

**Authors**: Fu-Chun Yang, Jason Eshraghian  

**Link**: [PDF](https://arxiv.org/pdf/2511.03945)  

**Abstract**: In multi-agent settings, such as debate, reflection, or tool-calling, large language models (LLMs) pass messages as plain tokens, discarding most latent semantics. This constrains information transfer and adds unnecessary computational overhead. We form a latent bridge via vector translations, which use learned mappings that enable direct semantic exchange between representation spaces. A dual-encoder translator trained between Llama-2-7B and Mistral-7B-Instruct attains an average cosine alignment of 0.538. Injecting the translated vectors at 30 percent blending strength steers the target model's generation without destabilizing logits. Bidirectional evaluation shows a 2.01:1 transfer asymmetry, indicating that general-purpose models yield more transferable representations than instruction-tuned variants. This conservative injection preserves computational stability while demonstrating that cross-model latent communication is feasible, enabling collaborative AI systems that share meaning rather than tokens. 

---
# The Human Flourishing Geographic Index: A County-Level Dataset for the United States, 2013--2023 

**Authors**: Stefano M. Iacus, Devika Jain, Andrea Nasuto, Giuseppe Porro, Marcello Carammia, Andrea Vezzulli  

**Link**: [PDF](https://arxiv.org/pdf/2511.03915)  

**Abstract**: Quantifying human flourishing, a multidimensional construct including happiness, health, purpose, virtue, relationships, and financial stability, is critical for understanding societal well-being beyond economic indicators. Existing measures often lack fine spatial and temporal resolution. Here we introduce the Human Flourishing Geographic Index (HFGI), derived from analyzing approximately 2.6 billion geolocated U.S. tweets (2013-2023) using fine-tuned large language models to classify expressions across 48 indicators aligned with Harvard's Global Flourishing Study framework plus attitudes towards migration and perception of corruption. The dataset offers monthly and yearly county- and state-level indicators of flourishing-related discourse, validated to confirm that the measures accurately represent the underlying constructs and show expected correlations with established indicators. This resource enables multidisciplinary analyses of well-being, inequality, and social change at unprecedented resolution, offering insights into the dynamics of human flourishing as reflected in social media discourse across the United States over the past decade. 

---
# Context informs pragmatic interpretation in vision-language models 

**Authors**: Alvin Wei Ming Tan, Ben Prystawski, Veronica Boyce, Michael C. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2511.03908)  

**Abstract**: Iterated reference games - in which players repeatedly pick out novel referents using language - present a test case for agents' ability to perform context-sensitive pragmatic reasoning in multi-turn linguistic environments. We tested humans and vision-language models on trials from iterated reference games, varying the given context in terms of amount, order, and relevance. Without relevant context, models were above chance but substantially worse than humans. However, with relevant context, model performance increased dramatically over trials. Few-shot reference games with abstract referents remain a difficult task for machine learning models. 

---
# GRAD: Graph-Retrieved Adaptive Decoding for Hallucination Mitigation 

**Authors**: Manh Nguyen, Sunil Gupta, Dai Do, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2511.03900)  

**Abstract**: Hallucination mitigation remains a persistent challenge for large language models (LLMs), even as model scales grow. Existing approaches often rely on external knowledge sources, such as structured databases or knowledge graphs, accessed through prompting or retrieval. However, prompt-based grounding is fragile and domain-sensitive, while symbolic knowledge integration incurs heavy retrieval and formatting costs. Motivated by knowledge graphs, we introduce Graph-Retrieved Adaptive Decoding (GRAD), a decoding-time method that grounds generation in corpus-derived evidence without retraining. GRAD constructs a sparse token transition graph by accumulating next-token logits across a small retrieved corpus in a single forward pass. During decoding, graph-retrieved logits are max-normalized and adaptively fused with model logits to favor high-evidence continuations while preserving fluency. Across three models and a range of question-answering benchmarks spanning intrinsic, extrinsic hallucination, and factuality tasks, GRAD consistently surpasses baselines, achieving up to 9.7$\%$ higher intrinsic accuracy, 8.6$\%$ lower hallucination rates, and 6.9$\%$ greater correctness compared to greedy decoding, while attaining the highest truth--informativeness product score among all methods. GRAD offers a lightweight, plug-and-play alternative to contrastive decoding and knowledge graph augmentation, demonstrating that statistical evidence from corpus-level token transitions can effectively steer generation toward more truthful and verifiable outputs. 

---
# Evaluating Machine Translation Datasets for Low-Web Data Languages: A Gendered Lens 

**Authors**: Hellina Hailu Nigatu, Bethelhem Yemane Mamo, Bontu Fufa Balcha, Debora Taye Tesfaye, Elbethel Daniel Zewdie, Ikram Behiru Nesiru, Jitu Ewnetu Hailu, Senait Mengesha Yayo  

**Link**: [PDF](https://arxiv.org/pdf/2511.03880)  

**Abstract**: As low-resourced languages are increasingly incorporated into NLP research, there is an emphasis on collecting large-scale datasets. But in prioritizing quantity over quality, we risk 1) building language technologies that perform poorly for these languages and 2) producing harmful content that perpetuates societal biases. In this paper, we investigate the quality of Machine Translation (MT) datasets for three low-resourced languages--Afan Oromo, Amharic, and Tigrinya, with a focus on the gender representation in the datasets. Our findings demonstrate that while training data has a large representation of political and religious domain text, benchmark datasets are focused on news, health, and sports. We also found a large skew towards the male gender--in names of persons, the grammatical gender of verbs, and in stereotypical depictions in the datasets. Further, we found harmful and toxic depictions against women, which were more prominent for the language with the largest amount of data, underscoring that quantity does not guarantee quality. We hope that our work inspires further inquiry into the datasets collected for low-resourced languages and prompts early mitigation of harmful content. WARNING: This paper contains discussion of NSFW content that some may find disturbing. 

---
# Divide, Cache, Conquer: Dichotomic Prompting for Efficient Multi-Label LLM-Based Classification 

**Authors**: Mikołaj Langner, Jan Eliasz, Ewa Rudnicka, Jan Kocoń  

**Link**: [PDF](https://arxiv.org/pdf/2511.03830)  

**Abstract**: We introduce a method for efficient multi-label text classification with large language models (LLMs), built on reformulating classification tasks as sequences of dichotomic (yes/no) decisions. Instead of generating all labels in a single structured response, each target dimension is queried independently, which, combined with a prefix caching mechanism, yields substantial efficiency gains for short-text inference without loss of accuracy. To demonstrate the approach, we focus on affective text analysis, covering 24 dimensions including emotions and sentiment. Using LLM-to-SLM distillation, a powerful annotator model (DeepSeek-V3) provides multiple annotations per text, which are aggregated to fine-tune smaller models (HerBERT-Large, CLARIN-1B, PLLuM-8B, Gemma3-1B). The fine-tuned models show significant improvements over zero-shot baselines, particularly on the dimensions seen during training. Our findings suggest that decomposing multi-label classification into dichotomic queries, combined with distillation and cache-aware inference, offers a scalable and effective framework for LLM-based classification. While we validate the method on affective states, the approach is general and applicable across domains. 

---
# STARS: Segment-level Token Alignment with Rejection Sampling in Large Language Models 

**Authors**: Mohammad Atif Quamar, Mohammad Areeb, Mikhail Kuznetsov, Muslum Ozgur Ozmen, Z. Berkay Celik  

**Link**: [PDF](https://arxiv.org/pdf/2511.03827)  

**Abstract**: Aligning large language models with human values is crucial for their safe deployment; however, existing methods, such as fine-tuning, are computationally expensive and suboptimal. In contrast, inference-time approaches like Best-of-N sampling require practically infeasible computation to achieve optimal alignment. We propose STARS: Segment-level Token Alignment with Rejection Sampling, a decoding-time algorithm that steers model generation by iteratively sampling, scoring, and rejecting/accepting short, fixed-size token segments. This allows for early correction of the generation path, significantly improving computational efficiency and boosting alignment quality. Across a suite of six LLMs, we show that STARS outperforms Supervised Fine-Tuning (SFT) by up to 14.9 percentage points and Direct Preference Optimization (DPO) by up to 4.3 percentage points on win-rates, while remaining highly competitive with strong Best-of-N baselines. Our work establishes granular, reward-guided sampling as a generalizable, robust, and efficient alternative to traditional fine-tuning and full-sequence ranking methods for aligning LLMs. 

---
# PLLuM: A Family of Polish Large Language Models 

**Authors**: Jan Kocoń, Maciej Piasecki, Arkadiusz Janz, Teddy Ferdinan, Łukasz Radliński, Bartłomiej Koptyra, Marcin Oleksy, Stanisław Woźniak, Paweł Walkowiak, Konrad Wojtasik, Julia Moska, Tomasz Naskręt, Bartosz Walkowiak, Mateusz Gniewkowski, Kamil Szyc, Dawid Motyka, Dawid Banach, Jonatan Dalasiński, Ewa Rudnicka, Bartłomiej Alberski, Tomasz Walkowiak, Aleksander Szczęsny, Maciej Markiewicz, Tomasz Bernaś, Hubert Mazur, Kamil Żyta, Mateusz Tykierko, Grzegorz Chodak, Tomasz Kajdanowicz, Przemysław Kazienko, Agnieszka Karlińska, Karolina Seweryn, Anna Kołos, Maciej Chrabąszcz, Katarzyna Lorenc, Aleksandra Krasnodębska, Artur Wilczek, Katarzyna Dziewulska, Paula Betscher, Zofia Cieślińska, Katarzyna Kowol, Daria Mikoś, Maciej Trzciński, Dawid Krutul, Marek Kozłowski, Sławomir Dadas, Rafał Poświata, Michał Perełkiewicz, Małgorzata Grębowiec, Maciej Kazuła, Marcin Białas, Roman Roszko, Danuta Roszko, Jurgita Vaičenonienė, Andrius Utka, Paweł Levchuk, Paweł Kowalski, Irena Prawdzic-Jankowska, Maciej Ogrodniczuk, Monika Borys, Anna Bulińska, Wiktoria Gumienna, Witold Kieraś, Dorota Komosińska, Katarzyna Krasnowska-Kieraś, Łukasz Kobyliński, Martyna Lewandowska, Marek Łaziński, Mikołaj Łątkowski, Dawid Mastalerz, Beata Milewicz, Agnieszka Anna Mykowiecka, Angelika Peljak-Łapińska, Sandra Penno, Zuzanna Przybysz, Michał Rudolf, Piotr Rybak, Karolina Saputa, Aleksandra Tomaszewska, Aleksander Wawer, Marcin Woliński, Joanna Wołoszyn, Alina Wróblewska, Bartosz Żuk, Filip Żarnecki, Konrad Kaczyński, Anna Cichosz, Zuzanna Deckert, Monika Garnys, Izabela Grabarczyk, Wojciech Janowski, Sylwia Karasińska, Aleksandra Kujawiak, Piotr Misztela, Maria Szymańska, Karolina Walkusz, Igor Siek, Jakub Kwiatkowski, Piotr Pęzik  

**Link**: [PDF](https://arxiv.org/pdf/2511.03823)  

**Abstract**: Large Language Models (LLMs) play a central role in modern artificial intelligence, yet their development has been primarily focused on English, resulting in limited support for other languages. We present PLLuM (Polish Large Language Model), the largest open-source family of foundation models tailored specifically for the Polish language. Developed by a consortium of major Polish research institutions, PLLuM addresses the need for high-quality, transparent, and culturally relevant language models beyond the English-centric commercial landscape. We describe the development process, including the construction of a new 140-billion-token Polish text corpus for pre-training, a 77k custom instructions dataset, and a 100k preference optimization dataset. A key component is a Responsible AI framework that incorporates strict data governance and a hybrid module for output correction and safety filtering. We detail the models' architecture, training procedures, and alignment techniques for both base and instruction-tuned variants, and demonstrate their utility in a downstream task within public administration. By releasing these models publicly, PLLuM aims to foster open research and strengthen sovereign AI technologies in Poland. 

---
# GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation 

**Authors**: Stergios Chatzikyriakidis, Dimitris Papadakis, Sevasti-Ioanna Papaioannou, Erofili Psaltaki  

**Link**: [PDF](https://arxiv.org/pdf/2511.03772)  

**Abstract**: We present an extended Greek Dialectal Dataset (GRDD+) 1that complements the existing GRDD dataset with more data from Cretan, Cypriot, Pontic and Northern Greek, while we add six new varieties: Greco-Corsican, Griko (Southern Italian Greek), Maniot, Heptanesian, Tsakonian, and Katharevusa Greek. The result is a dataset with total size 6,374,939 words and 10 varieties. This is the first dataset with such variation and size to date. We conduct a number of fine-tuning experiments to see the effect of good quality dialectal data on a number of LLMs. We fine-tune three model architectures (Llama-3-8B, Llama-3.1-8B, Krikri-8B) and compare the results to frontier models (Claude-3.7-Sonnet, Gemini-2.5, ChatGPT-5). 

---
# TextualVerifier: Verify TextGrad Step-by-Step 

**Authors**: Eugenius Mario Situmorang, Adila Alfa Krisnadhi, Ari Wibisono  

**Link**: [PDF](https://arxiv.org/pdf/2511.03739)  

**Abstract**: TextGrad is a novel approach to text-based automatic differentiation that enables composite AI systems to perform optimization without explicit numerical equations. However, it currently lacks self-verification mechanisms that ensure reasoning validity in text-based decision making. This research introduces TextualVerifier, a verification framework that leverages chain-of-thought reasoning and majority voting with large language models to address this verification gap. TextualVerifier implements a four-stage workflow: chain-of-thought decomposition, variant generation, majority voting, and consensus aggregation. It integrates non-invasively with TextGrad at both the loss function and optimization result verification stages. Experimental evaluation using the Gemini 1.5 Pro model is conducted in two phases: (1) standalone evaluation on PRM800K, and (2) integrated evaluation with TextGrad on GPQA-Diamond, MMLU-ML, and MMLU-CP benchmarks. Results show statistically significant improvements (p < 0.001). In phase one, TextualVerifier improves the validity of reasoning steps by 29 percent. In phase two, integration into TextGrad loss function yields a 2.2 percentage point gain from 68.2 to 70.4 percent with a moderate overhead of 5.9 LLM calls on average. Further evaluations of TextualVerifier versioning yield 8.08, 10.71, and 3.92 percentage point improvements on GPQA, MMLU-ML, and MMLU-CP respectively. TextualVerifier thus presents the first self-verification framework for TextGrad through LLM-based techniques without requiring numerical gradients, enabling more reliable reasoning and opening new directions for verification in text-based optimization. 

---
# Activation-Space Personality Steering: Hybrid Layer Selection for Stable Trait Control in LLMs 

**Authors**: Pranav Bhandari, Nicolas Fay, Sanjeevan Selvaganapathy, Amitava Datta, Usman Naseem, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2511.03738)  

**Abstract**: Large Language Models exhibit implicit personalities in their generation, but reliably controlling or aligning these traits to meet specific needs remains an open challenge. The need for effective mechanisms for behavioural manipulation of the model during generation is a critical gap in the literature that needs to be fulfilled. Personality-aware LLMs hold a promising direction towards this objective. However, the relationship between these psychological constructs and their representations within LLMs remains underexplored and requires further investigation. Moreover, it is intriguing to understand and study the use of these representations to steer the models' behaviour. We propose a novel pipeline that extracts hidden state activations from transformer layers using the Big Five Personality Traits (Openness, Conscientiousness, Extraversion, Agreeableness and Neuroticism), which is a comprehensive and empirically validated framework to model human personality applies low-rank subspace discovery methods, and identifies trait-specific optimal layers across different model architectures for robust injection. The resulting personality-aligned directions are then operationalised through a flexible steering framework with dynamic layer selection, enabling precise control of trait expression in LLM outputs. Our findings reveal that personality traits occupy a low-rank shared subspace, and that these latent structures can be transformed into actionable mechanisms for effective steering through careful perturbations without impacting the fluency, variance and general capabilities, helping to bridge the gap between psychological theory and practical model alignment. 

---
# VeriCoT: Neuro-symbolic Chain-of-Thought Validation via Logical Consistency Checks 

**Authors**: Yu Feng, Nathaniel Weir, Kaj Bostrom, Sam Bayless, Darion Cassel, Sapana Chaudhary, Benjamin Kiesl-Reiter, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2511.04662)  

**Abstract**: LLMs can perform multi-step reasoning through Chain-of-Thought (CoT), but they cannot reliably verify their own logic. Even when they reach correct answers, the underlying reasoning may be flawed, undermining trust in high-stakes scenarios. To mitigate this issue, we introduce VeriCoT, a neuro-symbolic method that extracts and verifies formal logical arguments from CoT reasoning. VeriCoT formalizes each CoT reasoning step into first-order logic and identifies premises that ground the argument in source context, commonsense knowledge, or prior reasoning steps. The symbolic representation enables automated solvers to verify logical validity while the NL premises allow humans and systems to identify ungrounded or fallacious reasoning steps. Experiments on the ProofWriter, LegalBench, and BioASQ datasets show VeriCoT effectively identifies flawed reasoning, and serves as a strong predictor of final answer correctness. We also leverage VeriCoT's verification signal for (1) inference-time self-reflection, (2) supervised fine-tuning (SFT) on VeriCoT-distilled datasets and (3) preference fine-tuning (PFT) with direct preference optimization (DPO) using verification-based pairwise rewards, further improving reasoning validity and accuracy. 

---
# DR. WELL: Dynamic Reasoning and Learning with Symbolic World Model for Embodied LLM-Based Multi-Agent Collaboration 

**Authors**: Narjes Nourzad, Hanqing Yang, Shiyu Chen, Carlee Joe-Wong  

**Link**: [PDF](https://arxiv.org/pdf/2511.04646)  

**Abstract**: Cooperative multi-agent planning requires agents to make joint decisions with partial information and limited communication. Coordination at the trajectory level often fails, as small deviations in timing or movement cascade into conflicts. Symbolic planning mitigates this challenge by raising the level of abstraction and providing a minimal vocabulary of actions that enable synchronization and collective progress. We present DR. WELL, a decentralized neurosymbolic framework for cooperative multi-agent planning. Cooperation unfolds through a two-phase negotiation protocol: agents first propose candidate roles with reasoning and then commit to a joint allocation under consensus and environment constraints. After commitment, each agent independently generates and executes a symbolic plan for its role without revealing detailed trajectories. Plans are grounded in execution outcomes via a shared world model that encodes the current state and is updated as agents act. By reasoning over symbolic plans rather than raw trajectories, DR. WELL avoids brittle step-level alignment and enables higher-level operations that are reusable, synchronizable, and interpretable. Experiments on cooperative block-push tasks show that agents adapt across episodes, with the dynamic world model capturing reusable patterns and improving task completion rates and efficiency. Experiments on cooperative block-push tasks show that our dynamic world model improves task completion and efficiency through negotiation and self-refinement, trading a time overhead for evolving, more efficient collaboration strategies. 

---
# Are We Asking the Right Questions? On Ambiguity in Natural Language Queries for Tabular Data Analysis 

**Authors**: Daniel Gomm, Cornelius Wolff, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2511.04584)  

**Abstract**: Natural language interfaces to tabular data must handle ambiguities inherent to queries. Instead of treating ambiguity as a deficiency, we reframe it as a feature of cooperative interaction, where the responsibility of query specification is shared among the user and the system. We develop a principled framework distinguishing cooperative queries, i.e., queries that yield a resolvable interpretation, from uncooperative queries that cannot be resolved. Applying the framework to evaluations for tabular question answering and analysis, we analyze the queries in 15 popular datasets, and observe an uncontrolled mixing of query types neither adequate for evaluating a system's execution accuracy nor for evaluating interpretation capabilities. Our framework and analysis of queries shifts the perspective from fixing ambiguity to embracing cooperation in resolving queries. This reflection enables more informed design and evaluation for natural language interfaces for tabular data, for which we outline implications and directions for future research. 

---
# Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper 

**Authors**: Atsuyuki Miyai, Mashiro Toyooka, Takashi Otonari, Zaiying Zhao, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2511.04583)  

**Abstract**: Understanding the current capabilities and risks of AI Scientist systems is essential for ensuring trustworthy and sustainable AI-driven scientific progress while preserving the integrity of the academic ecosystem. To this end, we develop Jr. AI Scientist, a state-of-the-art autonomous AI scientist system that mimics the core research workflow of a novice student researcher: Given the baseline paper from the human mentor, it analyzes its limitations, formulates novel hypotheses for improvement, validates them through rigorous experimentation, and writes a paper with the results. Unlike previous approaches that assume full automation or operate on small-scale code, Jr. AI Scientist follows a well-defined research workflow and leverages modern coding agents to handle complex, multi-file implementations, leading to scientifically valuable contributions. For evaluation, we conducted automated assessments using AI Reviewers, author-led evaluations, and submissions to Agents4Science, a venue dedicated to AI-driven scientific contributions. The findings demonstrate that Jr. AI Scientist generates papers receiving higher review scores than existing fully automated systems. Nevertheless, we identify important limitations from both the author evaluation and the Agents4Science reviews, indicating the potential risks of directly applying current AI Scientist systems and key challenges for future research. Finally, we comprehensively report various risks identified during development. We hope these insights will deepen understanding of current progress and risks in AI Scientist development. 

---
# Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm 

**Authors**: Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang, Xiaomeng Hu, Yining Zheng, Xinchi Chen, Jun Zhao, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04570)  

**Abstract**: "Thinking with Text" and "Thinking with Images" paradigm significantly improve the reasoning ability of large language models (LLMs) and Vision Language Models (VLMs). However, these paradigms have inherent limitations. (1) Images capture only single moments and fail to represent dynamic processes or continuous changes, and (2) The separation of text and vision as distinct modalities, hindering unified multimodal understanding and generation. To overcome these limitations, we introduce "Thinking with Video", a new paradigm that leverages video generation models, such as Sora-2, to bridge visual and textual reasoning in a unified temporal framework. To support this exploration, we developed the Video Thinking Benchmark (VideoThinkBench). VideoThinkBench encompasses two task categories: (1) vision-centric tasks (e.g., Eyeballing Puzzles), and (2) text-centric tasks (e.g., subsets of GSM8K, MMMU). Our evaluation establishes Sora-2 as a capable reasoner. On vision-centric tasks, Sora-2 is generally comparable to state-of-the-art (SOTA) VLMs, and even surpasses VLMs on several tasks, such as Eyeballing Games. On text-centric tasks, Sora-2 achieves 92% accuracy on MATH, and 75.53% accuracy on MMMU. Furthermore, we systematically analyse the source of these abilities. We also find that self-consistency and in-context learning can improve Sora-2's performance. In summary, our findings demonstrate that the video generation model is the potential unified multimodal understanding and generation model, positions "thinking with video" as a unified multimodal reasoning paradigm. 

---
# Large language models replicate and predict human cooperation across experiments in game theory 

**Authors**: Andrea Cera Palatsi, Samuel Martin-Gutierrez, Ana S. Cardenal, Max Pellert  

**Link**: [PDF](https://arxiv.org/pdf/2511.04500)  

**Abstract**: Large language models (LLMs) are increasingly used both to make decisions in domains such as health, education and law, and to simulate human behavior. Yet how closely LLMs mirror actual human decision-making remains poorly understood. This gap is critical: misalignment could produce harmful outcomes in practical applications, while failure to replicate human behavior renders LLMs ineffective for social simulations. Here, we address this gap by developing a digital twin of game-theoretic experiments and introducing a systematic prompting and probing framework for machine-behavioral evaluation. Testing three open-source models (Llama, Mistral and Qwen), we find that Llama reproduces human cooperation patterns with high fidelity, capturing human deviations from rational choice theory, while Qwen aligns closely with Nash equilibrium predictions. Notably, we achieved population-level behavioral replication without persona-based prompting, simplifying the simulation process. Extending beyond the original human-tested games, we generate and preregister testable hypotheses for novel game configurations outside the original parameter grid. Our findings demonstrate that appropriately calibrated LLMs can replicate aggregate human behavioral patterns and enable systematic exploration of unexplored experimental spaces, offering a complementary approach to traditional research in the social and behavioral sciences that generates new empirical predictions about human social decision-making. 

---
# Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs 

**Authors**: Alberto Cattaneo, Carlo Luschi, Daniel Justus  

**Link**: [PDF](https://arxiv.org/pdf/2511.04473)  

**Abstract**: Retrieval of information from graph-structured knowledge bases represents a promising direction for improving the factuality of LLMs. While various solutions have been proposed, a comparison of methods is difficult due to the lack of challenging QA datasets with ground-truth targets for graph retrieval. We present SynthKGQA, a framework for generating high-quality synthetic Knowledge Graph Question Answering datasets from any Knowledge Graph, providing the full set of ground-truth facts in the KG to reason over each question. We show how, in addition to enabling more informative benchmarking of KG retrievers, the data produced with SynthKGQA also allows us to train better models. We apply SynthKGQA to Wikidata to generate GTSQA, a new dataset designed to test zero-shot generalization abilities of KG retrievers with respect to unseen graph structures and relation types, and benchmark popular solutions for KG-augmented LLMs on it. 

---
# The Illusion of Certainty: Uncertainty quantification for LLMs fails under ambiguity 

**Authors**: Tim Tomov, Dominik Fuchsgruber, Tom Wollschläger, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2511.04418)  

**Abstract**: Accurate uncertainty quantification (UQ) in Large Language Models (LLMs) is critical for trustworthy deployment. While real-world language is inherently ambiguous, reflecting aleatoric uncertainty, existing UQ methods are typically benchmarked against tasks with no ambiguity. In this work, we demonstrate that while current uncertainty estimators perform well under the restrictive assumption of no ambiguity, they degrade to close-to-random performance on ambiguous data. To this end, we introduce MAQA* and AmbigQA*, the first ambiguous question-answering (QA) datasets equipped with ground-truth answer distributions estimated from factual co-occurrence. We find this performance deterioration to be consistent across different estimation paradigms: using the predictive distribution itself, internal representations throughout the model, and an ensemble of models. We show that this phenomenon can be theoretically explained, revealing that predictive-distribution and ensemble-based estimators are fundamentally limited under ambiguity. Overall, our study reveals a key shortcoming of current UQ methods for LLMs and motivates a rethinking of current modeling paradigms. 

---
# Black-Box Guardrail Reverse-engineering Attack 

**Authors**: Hongwei Yao, Yun Xia, Shuo Shao, Haoran Shi, Tong Qiao, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04215)  

**Abstract**: Large language models (LLMs) increasingly employ guardrails to enforce ethical, legal, and application-specific constraints on their outputs. While effective at mitigating harmful responses, these guardrails introduce a new class of vulnerabilities by exposing observable decision patterns. In this work, we present the first study of black-box LLM guardrail reverse-engineering attacks. We propose Guardrail Reverse-engineering Attack (GRA), a reinforcement learning-based framework that leverages genetic algorithm-driven data augmentation to approximate the decision-making policy of victim guardrails. By iteratively collecting input-output pairs, prioritizing divergence cases, and applying targeted mutations and crossovers, our method incrementally converges toward a high-fidelity surrogate of the victim guardrail. We evaluate GRA on three widely deployed commercial systems, namely ChatGPT, DeepSeek, and Qwen3, and demonstrate that it achieves an rule matching rate exceeding 0.92 while requiring less than $85 in API costs. These findings underscore the practical feasibility of guardrail extraction and highlight significant security risks for current LLM safety mechanisms. Our findings expose critical vulnerabilities in current guardrail designs and highlight the urgent need for more robust defense mechanisms in LLM deployment. 

---
# Block Rotation is All You Need for MXFP4 Quantization 

**Authors**: Yuantian Shao, Peisong Wang, Yuanteng Chen, Chang Xu, Zhihui Wei, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.04214)  

**Abstract**: Large language models (LLMs) have achieved remarkable success, but their rapidly growing scale imposes prohibitive costs in memory, computation, and energy. Post-training quantization (PTQ) is a promising solution for efficient deployment, yet achieving accurate W4A4 quantization remains an open challenge. While most existing methods are designed for INT4 formats, the emergence of MXFP4 -- a new FP4 format with various hardware support (NVIDIA, AMD, Intel)-- raises questions about the applicability of current techniques. In this work, we establish a comprehensive benchmark of PTQ methods under the MXFP4 format. Through systematic evaluation, we find that methods like GPTQ consistently deliver strong performance, whereas rotation-based approaches, which are almost used by all state-of-the-art approaches, suffer from severe incompatibility with MXFP4. We further provide the first in-depth analysis of this conflict, tracing its root to a fundamental mismatch between MXFP4's PoT (power-of-two) block scaling and the redistribution of outlier energy via global rotation. Building on this insight, we propose a simple yet effective block rotation strategy that adapts rotation-based methods to MXFP4, leading to substantial accuracy improvements across diverse LLMs. Our findings not only offer clear guidance for practitioners but also set a foundation for advancing PTQ research under emerging low-precision formats. 

---
# Transforming Mentorship: An AI Powered Chatbot Approach to University Guidance 

**Authors**: Mashrur Rahman, Mantaqa abedin, Monowar Zamil Abir, Faizul Islam Ansari, Adib Reza, Farig Yousuf Sadeque, Niloy Farhan  

**Link**: [PDF](https://arxiv.org/pdf/2511.04172)  

**Abstract**: University students face immense challenges during their undergraduate lives, often being deprived of personalized on-demand guidance that mentors fail to provide at scale. Digital tools exist, but there is a serious lack of customized coaching for newcomers. This paper presents an AI-powered chatbot that will serve as a mentor for the students of BRAC University. The main component is a data ingestion pipeline that efficiently processes and updates information from diverse sources, such as CSV files and university webpages. The chatbot retrieves information through a hybrid approach, combining BM25 lexical ranking with ChromaDB semantic retrieval, and uses a Large Language Model, LLaMA-3.3-70B, to generate conversational responses. The generated text was found to be semantically highly relevant, with a BERTScore of 0.831 and a METEOR score of 0.809. The data pipeline was also very efficient, taking 106.82 seconds for updates, compared to 368.62 seconds for new data. This chatbot will be able to help students by responding to their queries, helping them to get a better understanding of university life, and assisting them to plan better routines for their semester in the open-credit university. 

---
# Seeing Straight: Document Orientation Detection for Efficient OCR 

**Authors**: Suranjan Goswami, Abhinav Ravi, Raja Kolla, Ali Faraz, Shaharukh Khan, Akash, Chandra Khatri, Shubham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2511.04161)  

**Abstract**: Despite significant advances in document understanding, determining the correct orientation of scanned or photographed documents remains a critical pre-processing step in the real world settings. Accurate rotation correction is essential for enhancing the performance of downstream tasks such as Optical Character Recognition (OCR) where misalignment commonly arises due to user errors, particularly incorrect base orientations of the camera during capture. In this study, we first introduce OCR-Rotation-Bench (ORB), a new benchmark for evaluating OCR robustness to image rotations, comprising (i) ORB-En, built from rotation-transformed structured and free-form English OCR datasets, and (ii) ORB-Indic, a novel multilingual set spanning 11 Indic mid to low-resource languages. We also present a fast, robust and lightweight rotation classification pipeline built on the vision encoder of Phi-3.5-Vision model with dynamic image cropping, fine-tuned specifically for 4-class rotation task in a standalone fashion. Our method achieves near-perfect 96% and 92% accuracy on identifying the rotations respectively on both the datasets. Beyond classification, we demonstrate the critical role of our module in boosting OCR performance: closed-source (up to 14%) and open-weights models (up to 4x) in the simulated real-world setting. 

---
# Sub-exponential Growth in Online Word Usage: A Piecewise Power-Law Model 

**Authors**: Hayafumi Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2511.04106)  

**Abstract**: The diffusion of ideas and language in society has conventionally been described by S-shaped models, such as the logistic curve. However, the role of sub-exponential growth -a slower than exponential pattern known in epidemiology- has been largely overlooked in broader social phenomena. Here, we present a piecewise power-law model to characterize complex growth curves with a few parameters. We systematically analyzed a large-scale dataset of approximately one billion Japanese blog articles linked to Wikipedia vocabulary, and observed consistent patterns in web search trend data (English, Spanish, and Japanese). Our analysis of the 2,965 selected items reveals that about 55% (1,625 items) were found to have no abrupt jumps and were well captured by one or two segments. For single-segment curves, we found that (i) the mode of the shape parameter alpha was near 0.5, indicating prevalent sub-exponential growth; (ii) the ultimate diffusion scale is primarily determined by the growth rate R, with minor contributions from alpha or the duration T; and (iii) alpha showed a tendency to vary with the nature of the topic, being smaller for niche/local topics and larger for widely shared ones. Furthermore, a micro-behavioral model distinguishing outward contact with strangers from inward interaction within their community suggests that alpha can be interpreted as an index of the preference for outward-oriented communication. These findings suggest that sub-exponential growth is a common pattern of social diffusion, and our model provides a practical framework for consistently describing, comparing, and interpreting complex and diverse growth curves. 

---
# DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization 

**Authors**: Yuantian Shao, Yuanteng Chen, Peisong Wang, Jianlin Yu, Jing Lin, Yiwu Yao, Zhihui Wei, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.04063)  

**Abstract**: Quantization plays a crucial role in accelerating the inference of large-scale models, and rotational matrices have been shown to effectively improve quantization performance by smoothing outliers. However, end-to-end fine-tuning of rotational optimization algorithms incurs high computational costs and is prone to overfitting. To address this challenge, we propose an efficient distribution-aware rotational calibration method, DartQuant, which reduces the complexity of rotational optimization by constraining the distribution of the activations after rotation. This approach also effectively reduces reliance on task-specific losses, thereby mitigating the risk of overfitting. Additionally, we introduce the QR-Orth optimization scheme, which replaces expensive alternating optimization with a more efficient solution. In a variety of model quantization experiments, DartQuant demonstrates superior performance. Compared to existing methods, it achieves 47$\times$ acceleration and 10$\times$ memory savings for rotational optimization on a 70B model. Furthermore, it is the first to successfully complete rotational calibration for a 70B model on a single 3090 GPU, making quantization of large language models feasible in resource-constrained environments. Code is available at this https URL. 

---
# Explorability in Pushdown Automata 

**Authors**: Ayaan Bedi, Karoliina Lehtinen  

**Link**: [PDF](https://arxiv.org/pdf/2511.04048)  

**Abstract**: We study explorability, a measure of nondeterminism in pushdown automata, which generalises history-determinism. An automaton is k-explorable if, while reading the input, it suffices to follow k concurrent runs, built step-by-step based only on the input seen so far, to construct an accepting one, if it exists. We show that the class of explorable PDAs lies strictly between history-deterministic and fully nondeterministic PDAs in terms of both expressiveness and succinctness. In fact increasing explorability induces an infinite hierarchy: each level k defines a strictly more expressive class than level k-1, yet the entire class remains less expressive than general nondeterministic PDAs. We then introduce a parameterized notion of explorability, where the number of runs may depend on input length, and show that exponential explorability precisely captures the context-free languages. Finally, we prove that explorable PDAs can be doubly exponentially more succinct than history-deterministic ones, and that the succinctness gap between deterministic and 2-explorable PDAs is not recursively enumerable. These results position explorability as a robust and operationally meaningful measure of nondeterminism for pushdown systems. 

---
# Towards Scalable Meta-Learning of near-optimal Interpretable Models via Synthetic Model Generations 

**Authors**: Kyaw Hpone Myint, Zhe Wu, Alexandre G.R. Day, Giri Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2511.04000)  

**Abstract**: Decision trees are widely used in high-stakes fields like finance and healthcare due to their interpretability. This work introduces an efficient, scalable method for generating synthetic pre-training data to enable meta-learning of decision trees. Our approach samples near-optimal decision trees synthetically, creating large-scale, realistic datasets. Using the MetaTree transformer architecture, we demonstrate that this method achieves performance comparable to pre-training on real-world data or with computationally expensive optimal decision trees. This strategy significantly reduces computational costs, enhances data generation flexibility, and paves the way for scalable and efficient meta-learning of interpretable decision tree models. 

---
# LLMs and Cultural Values: the Impact of Prompt Language and Explicit Cultural Framing 

**Authors**: Bram Bulté, Ayla Rigouts Terryn  

**Link**: [PDF](https://arxiv.org/pdf/2511.03980)  

**Abstract**: Large Language Models (LLMs) are rapidly being adopted by users across the globe, who interact with them in a diverse range of languages. At the same time, there are well-documented imbalances in the training data and optimisation objectives of this technology, raising doubts as to whether LLMs can represent the cultural diversity of their broad user base. In this study, we look at LLMs and cultural values and examine how prompt language and cultural framing influence model responses and their alignment with human values in different countries. We probe 10 LLMs with 63 items from the Hofstede Values Survey Module and World Values Survey, translated into 11 languages, and formulated as prompts with and without different explicit cultural perspectives. Our study confirms that both prompt language and cultural perspective produce variation in LLM outputs, but with an important caveat: While targeted prompting can, to a certain extent, steer LLM responses in the direction of the predominant values of the corresponding countries, it does not overcome the models' systematic bias toward the values associated with a restricted set of countries in our dataset: the Netherlands, Germany, the US, and Japan. All tested models, regardless of their origin, exhibit remarkably similar patterns: They produce fairly neutral responses on most topics, with selective progressive stances on issues such as social tolerance. Alignment with cultural values of human respondents is improved more with an explicit cultural perspective than with a targeted prompt language. Unexpectedly, combining both approaches is no more effective than cultural framing with an English prompt. These findings reveal that LLMs occupy an uncomfortable middle ground: They are responsive enough to changes in prompts to produce variation, but too firmly anchored to specific cultural defaults to adequately represent cultural diversity. 

---
# Multi-Agent Collaborative Framework For Math Problem Generation 

**Authors**: Kia Karbasi, Kevin Hong, Mohammad Amin Samadi, Gregory Pottie  

**Link**: [PDF](https://arxiv.org/pdf/2511.03958)  

**Abstract**: Automatic question generation (AQG) for mathematics education remains an elusive goal for Intelligent Tutoring Systems and educators. While pre-trained transformer-based language models have significantly advanced natural language generation, they often struggle to precisely control problem complexity and cognitive demands. In this paper, we introduce a collaborative multi-agent framework as a novel method of incorporating inference-time computation into AQG. This approach leverages multiple agents that iteratively refine generated question-answer pairs to better balance complexity and cognitive demand. We evaluate the generated questions on five meta-evaluation criteria: relevance, importance, clarity, difficulty matching, answerability, to assess the system's ability to control the required complexity and quality of the questions. Preliminary evaluations show that this collaborative multi-agent framework elevates the quality of generated educational content by fostering a more nuanced balance between cognitive challenge and clarity. These promising outcomes suggest that integrating collaborative multi-agent workflows can yield more controlled, pedagogically valuable content that can help advance automated educational content generation and adaptive learning environments. 

---
# MIDI-LLM: Adapting Large Language Models for Text-to-MIDI Music Generation 

**Authors**: Shih-Lun Wu, Yoon Kim, Cheng-Zhi Anna Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.03942)  

**Abstract**: We present MIDI-LLM, an LLM for generating multitrack MIDI music from free-form text prompts. Our approach expands a text LLM's vocabulary to include MIDI tokens, and uses a two-stage training recipe to endow text-to-MIDI abilities. By preserving the original LLM's parameter structure, we can directly leverage the vLLM library for accelerated inference. Experiments show that MIDI-LLM achieves higher quality, better text control, and faster inference compared to the recent Text2midi model. Live demo at this https URL. 

---
# RLHF: A comprehensive Survey for Cultural, Multimodal and Low Latency Alignment Methods 

**Authors**: Raghav Sharma, Manan Mehta, Sai Tiger Raina  

**Link**: [PDF](https://arxiv.org/pdf/2511.03939)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is the standard for aligning Large Language Models (LLMs), yet recent progress has moved beyond canonical text-based methods. This survey synthesizes the new frontier of alignment research by addressing critical gaps in multi-modal alignment, cultural fairness, and low-latency optimization. To systematically explore these domains, we first review foundational algo- rithms, including PPO, DPO, and GRPO, before presenting a detailed analysis of the latest innovations. By providing a comparative synthesis of these techniques and outlining open challenges, this work serves as an essential roadmap for researchers building more robust, efficient, and equitable AI systems. 

---
# How Different Tokenization Algorithms Impact LLMs and Transformer Models for Binary Code Analysis 

**Authors**: Ahmed Mostafa, Raisul Arefin Nahid, Samuel Mulder  

**Link**: [PDF](https://arxiv.org/pdf/2511.03825)  

**Abstract**: Tokenization is fundamental in assembly code analysis, impacting intrinsic characteristics like vocabulary size, semantic coverage, and extrinsic performance in downstream tasks. Despite its significance, tokenization in the context of assembly code remains an underexplored area. This study aims to address this gap by evaluating the intrinsic properties of Natural Language Processing (NLP) tokenization models and parameter choices, such as vocabulary size. We explore preprocessing customization options and pre-tokenization rules tailored to the unique characteristics of assembly code. Additionally, we assess their impact on downstream tasks like function signature prediction -- a critical problem in binary code analysis.
To this end, we conduct a thorough study on various tokenization models, systematically analyzing their efficiency in encoding assembly instructions and capturing semantic nuances. Through intrinsic evaluations, we compare tokenizers based on tokenization efficiency, vocabulary compression, and representational fidelity for assembly code. Using state-of-the-art pre-trained models such as the decoder-only Large Language Model (LLM) Llama 3.2, the encoder-only transformer BERT, and the encoder-decoder model BART, we evaluate the effectiveness of these tokenizers across multiple performance metrics. Preliminary findings indicate that tokenizer choice significantly influences downstream performance, with intrinsic metrics providing partial but incomplete predictability of extrinsic evaluation outcomes. These results reveal complex trade-offs between intrinsic tokenizer properties and their utility in practical assembly code tasks. Ultimately, this study provides valuable insights into optimizing tokenization models for low-level code analysis, contributing to the robustness and scalability of Natural Language Model (NLM)-based binary analysis workflows. 

---
# MimiTalk: Revolutionizing Qualitative Research with Dual-Agent AI 

**Authors**: Fengming Liu, Shubin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.03731)  

**Abstract**: We present MimiTalk, a dual-agent constitutional AI framework designed for scalable and ethical conversational data collection in social science research. The framework integrates a supervisor model for strategic oversight and a conversational model for question generation. We conducted three studies: Study 1 evaluated usability with 20 participants; Study 2 compared 121 AI interviews to 1,271 human interviews from the MediaSum dataset using NLP metrics and propensity score matching; Study 3 involved 10 interdisciplinary researchers conducting both human and AI interviews, followed by blind thematic analysis. Results across studies indicate that MimiTalk reduces interview anxiety, maintains conversational coherence, and outperforms human interviews in information richness, coherence, and stability. AI interviews elicit technical insights and candid views on sensitive topics, while human interviews better capture cultural and emotional nuances. These findings suggest that dual-agent constitutional AI supports effective human-AI collaboration, enabling replicable, scalable and quality-controlled qualitative research. 

---
