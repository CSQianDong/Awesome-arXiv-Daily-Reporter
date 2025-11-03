# Continuous Autoregressive Language Models 

**Authors**: Chenze Shao, Darren Li, Fandong Meng, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.27688)  

**Abstract**: The efficiency of large language models (LLMs) is fundamentally limited by their sequential, token-by-token generation process. We argue that overcoming this bottleneck requires a new design axis for LLM scaling: increasing the semantic bandwidth of each generative step. To this end, we introduce Continuous Autoregressive Language Models (CALM), a paradigm shift from discrete next-token prediction to continuous next-vector prediction. CALM uses a high-fidelity autoencoder to compress a chunk of K tokens into a single continuous vector, from which the original tokens can be reconstructed with over 99.9\% accuracy. This allows us to model language as a sequence of continuous vectors instead of discrete tokens, which reduces the number of generative steps by a factor of K. The paradigm shift necessitates a new modeling toolkit; therefore, we develop a comprehensive likelihood-free framework that enables robust training, evaluation, and controllable sampling in the continuous domain. Experiments show that CALM significantly improves the performance-compute trade-off, achieving the performance of strong discrete baselines at a significantly lower computational cost. More importantly, these findings establish next-vector prediction as a powerful and scalable pathway towards ultra-efficient language models. Code: this https URL. Project: this https URL. 

---
# Culture Cartography: Mapping the Landscape of Cultural Knowledge 

**Authors**: Caleb Ziems, William Held, Jane Yu, Amir Goldberg, David Grusky, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27672)  

**Abstract**: To serve global users safely and productively, LLMs need culture-specific knowledge that might not be learned during pre-training. How do we find such knowledge that is (1) salient to in-group users, but (2) unknown to LLMs? The most common solutions are single-initiative: either researchers define challenging questions that users passively answer (traditional annotation), or users actively produce data that researchers structure as benchmarks (knowledge extraction). The process would benefit from mixed-initiative collaboration, where users guide the process to meaningfully reflect their cultures, and LLMs steer the process towards more challenging questions that meet the researcher's goals. We propose a mixed-initiative methodology called CultureCartography. Here, an LLM initializes annotation with questions for which it has low-confidence answers, making explicit both its prior knowledge and the gaps therein. This allows a human respondent to fill these gaps and steer the model towards salient topics through direct edits. We implement this methodology as a tool called CultureExplorer. Compared to a baseline where humans answer LLM-proposed questions, we find that CultureExplorer more effectively produces knowledge that leading models like DeepSeek R1 and GPT-4o are missing, even with web search. Fine-tuning on this data boosts the accuracy of Llama-3.1-8B by up to 19.2% on related culture benchmarks. 

---
# SpecAttn: Speculating Sparse Attention 

**Authors**: Harsh Shah  

**Link**: [PDF](https://arxiv.org/pdf/2510.27641)  

**Abstract**: Large Language Models (LLMs) face significant computational bottlenecks during inference due to the quadratic complexity of self-attention mechanisms, particularly as context lengths increase. We introduce SpecAttn, a novel training-free approach that seamlessly integrates with existing speculative decoding techniques to enable efficient sparse attention in pre-trained transformers. Our key insight is to exploit the attention weights already computed by the draft model during speculative decoding to identify important tokens for the target model, eliminating redundant computation while maintaining output quality. SpecAttn employs three core techniques: KL divergence-based layer alignment between draft and target models, a GPU-optimized sorting-free algorithm for top-p token selection from draft attention patterns, and dynamic key-value cache pruning guided by these predictions. By leveraging the computational work already performed in standard speculative decoding pipelines, SpecAttn achieves over 75% reduction in key-value cache accesses with a mere 15.29% increase in perplexity on the PG-19 dataset, significantly outperforming existing sparse attention methods. Our approach demonstrates that speculative execution can be enhanced to provide approximate verification without significant performance degradation. 

---
# MARAG-R1: Beyond Single Retriever via Reinforcement-Learned Multi-Tool Agentic Retrieval 

**Authors**: Qi Luo, Xiaonan Li, Yuxin Wang, Tingshuo Fan, Yuan Li, Xinchi Chen, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27569)  

**Abstract**: Large Language Models (LLMs) excel at reasoning and generation but are inherently limited by static pretraining data, resulting in factual inaccuracies and weak adaptability to new information. Retrieval-Augmented Generation (RAG) addresses this issue by grounding LLMs in external knowledge; However, the effectiveness of RAG critically depends on whether the model can adequately access relevant information. Existing RAG systems rely on a single retriever with fixed top-k selection, restricting access to a narrow and static subset of the corpus. As a result, this single-retriever paradigm has become the primary bottleneck for comprehensive external information acquisition, especially in tasks requiring corpus-level reasoning. To overcome this limitation, we propose MARAG-R1, a reinforcement-learned multi-tool RAG framework that enables LLMs to dynamically coordinate multiple retrieval mechanisms for broader and more precise information access. MARAG-R1 equips the model with four retrieval tools -- semantic search, keyword search, filtering, and aggregation -- and learns both how and when to use them through a two-stage training process: supervised fine-tuning followed by reinforcement learning. This design allows the model to interleave reasoning and retrieval, progressively gathering sufficient evidence for corpus-level synthesis. Experiments on GlobalQA, HotpotQA, and 2WikiMultiHopQA demonstrate that MARAG-R1 substantially outperforms strong baselines and achieves new state-of-the-art results in corpus-level reasoning tasks. 

---
# Data-Efficient Domain Adaptation for LLM-based MT using Contrastive Preference Optimization 

**Authors**: Inacio Vieira, Antonio Castaldo, James O'Doherty, Sheila Castilho  

**Link**: [PDF](https://arxiv.org/pdf/2510.27556)  

**Abstract**: LLMs often require adaptation to domain-specific requirements, a process that can be expensive when relying solely on SFT. We present an empirical study on applying CPO to simulate a post-editing workflow for data-efficient domain adaptation. Our approach synthesizes preference pairs by treating the base model's own raw output as the 'rejected' translation and the human-approved TM entry as the 'chosen' one. This method provides direct feedback on the model's current knowledge, guiding it to align with domain-specific standards. Experiments in English-Brazilian Portuguese and English-Korean show that, by using just 14.7k preference pairs, the model achieves performance close to that of a model trained on 160k+ samples with SFT, demonstrating significant data efficiency. Although we showcase its effectiveness in MT, this application of CPO naturally generalizes to other generative tasks where a model's initial drafts can serve as a contrastive signal against a golden reference. 

---
# Multilingual BERT language model for medical tasks: Evaluation on domain-specific adaptation and cross-linguality 

**Authors**: Yinghao Luo, Lang Zhou, Amrish Jhingoer, Klaske Vliegenthart Jongbloed, Carlijn Jordans, Ben Werkhoven, Tom Seinen, Erik van Mulligen, Casper Rokx, Yunlei Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.27552)  

**Abstract**: In multilingual healthcare applications, the availability of domain-specific natural language processing(NLP) tools is limited, especially for low-resource languages. Although multilingual bidirectional encoder representations from transformers (BERT) offers a promising motivation to mitigate the language gap, the medical NLP tasks in low-resource languages are still underexplored. Therefore, this study investigates how further pre-training on domain-specific corpora affects model performance on medical tasks, focusing on three languages: Dutch, Romanian and Spanish. In terms of further pre-training, we conducted four experiments to create medical domain models. Then, these models were fine-tuned on three downstream tasks: Automated patient screening in Dutch clinical notes, named entity recognition in Romanian and Spanish clinical notes. Results show that domain adaptation significantly enhanced task performance. Furthermore, further differentiation of domains, e.g. clinical and general biomedical domains, resulted in diverse performances. The clinical domain-adapted model outperformed the more general biomedical domain-adapted model. Moreover, we observed evidence of cross-lingual transferability. Moreover, we also conducted further investigations to explore potential reasons contributing to these performance differences. These findings highlight the feasibility of domain adaptation and cross-lingual ability in medical NLP. Within the low-resource language settings, these findings can provide meaningful guidance for developing multilingual medical NLP systems to mitigate the lack of training data and thereby improve the model performance. 

---
# DialectalArabicMMLU: Benchmarking Dialectal Capabilities in Arabic and Multilingual Language Models 

**Authors**: Malik H. Altakrori, Nizar Habash, Abdelhakim Freihat, Younes Samih, Kirill Chirkunov, Muhammed AbuOdeh, Radu Florian, Teresa Lynn, Preslav Nakov, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2510.27543)  

**Abstract**: We present DialectalArabicMMLU, a new benchmark for evaluating the performance of large language models (LLMs) across Arabic dialects. While recently developed Arabic and multilingual benchmarks have advanced LLM evaluation for Modern Standard Arabic (MSA), dialectal varieties remain underrepresented despite their prevalence in everyday communication. DialectalArabicMMLU extends the MMLU-Redux framework through manual translation and adaptation of 3K multiple-choice question-answer pairs into five major dialects (Syrian, Egyptian, Emirati, Saudi, and Moroccan), yielding a total of 15K QA pairs across 32 academic and professional domains (22K QA pairs when also including English and MSA). The benchmark enables systematic assessment of LLM reasoning and comprehension beyond MSA, supporting both task-based and linguistic analysis. We evaluate 19 open-weight Arabic and multilingual LLMs (1B-13B parameters) and report substantial performance variation across dialects, revealing persistent gaps in dialectal generalization. DialectalArabicMMLU provides the first unified, human-curated resource for measuring dialectal understanding in Arabic, thus promoting more inclusive evaluation and future model development. 

---
# Patient-Centered Summarization Framework for AI Clinical Summarization: A Mixed-Methods Design 

**Authors**: Maria Lizarazo Jimenez, Ana Gabriela Claros, Kieran Green, David Toro-Tobon, Felipe Larios, Sheena Asthana, Camila Wenczenovicz, Kerly Guevara Maldonado, Luis Vilatuna-Andrango, Cristina Proano-Velez, Satya Sai Sri Bandi, Shubhangi Bagewadi, Megan E. Branda, Misk Al Zahidy, Saturnino Luz, Mirella Lapata, Juan P. Brito, Oscar J. Ponce-Ponte  

**Link**: [PDF](https://arxiv.org/pdf/2510.27535)  

**Abstract**: Large Language Models (LLMs) are increasingly demonstrating the potential to reach human-level performance in generating clinical summaries from patient-clinician conversations. However, these summaries often focus on patients' biology rather than their preferences, values, wishes, and concerns. To achieve patient-centered care, we propose a new standard for Artificial Intelligence (AI) clinical summarization tasks: Patient-Centered Summaries (PCS). Our objective was to develop a framework to generate PCS that capture patient values and ensure clinical utility and to assess whether current open-source LLMs can achieve human-level performance in this task. We used a mixed-methods process. Two Patient and Public Involvement groups (10 patients and 8 clinicians) in the United Kingdom participated in semi-structured interviews exploring what personal and contextual information should be included in clinical summaries and how it should be structured for clinical use. Findings informed annotation guidelines used by eight clinicians to create gold-standard PCS from 88 atrial fibrillation consultations. Sixteen consultations were used to refine a prompt aligned with the guidelines. Five open-source LLMs (Llama-3.2-3B, Llama-3.1-8B, Mistral-8B, Gemma-3-4B, and Qwen3-8B) generated summaries for 72 consultations using zero-shot and few-shot prompting, evaluated with ROUGE-L, BERTScore, and qualitative metrics. Patients emphasized lifestyle routines, social support, recent stressors, and care values. Clinicians sought concise functional, psychosocial, and emotional context. The best zero-shot performance was achieved by Mistral-8B (ROUGE-L 0.189) and Llama-3.1-8B (BERTScore 0.673); the best few-shot by Llama-3.1-8B (ROUGE-L 0.206, BERTScore 0.683). Completeness and fluency were similar between experts and models, while correctness and patient-centeredness favored human PCS. 

---
# SQLSpace: A Representation Space for Text-to-SQL to Discover and Mitigate Robustness Gaps 

**Authors**: Neha Srikanth, Victor Bursztyn, Puneet Mathur, Ani Nenkova  

**Link**: [PDF](https://arxiv.org/pdf/2510.27532)  

**Abstract**: We introduce SQLSpace, a human-interpretable, generalizable, compact representation for text-to-SQL examples derived with minimal human intervention. We demonstrate the utility of these representations in evaluation with three use cases: (i) closely comparing and contrasting the composition of popular text-to-SQL benchmarks to identify unique dimensions of examples they evaluate, (ii) understanding model performance at a granular level beyond overall accuracy scores, and (iii) improving model performance through targeted query rewriting based on learned correctness estimation. We show that SQLSpace enables analysis that would be difficult with raw examples alone: it reveals compositional differences between benchmarks, exposes performance patterns obscured by accuracy alone, and supports modeling of query success. 

---
# BiSparse-AAS: Bilinear Sparse Attention and Adaptive Spans Framework for Scalable and Efficient Text Summarization 

**Authors**: Desta Haileselassie Hagos, Legand L. Burge, Anietie Andy, Anis Yazidi, Vladimir Vlassov  

**Link**: [PDF](https://arxiv.org/pdf/2510.27516)  

**Abstract**: Transformer-based architectures have advanced text summarization, yet their quadratic complexity limits scalability on long documents. This paper introduces BiSparse-AAS (Bilinear Sparse Attention with Adaptive Spans), a novel framework that combines sparse attention, adaptive spans, and bilinear attention to address these limitations. Sparse attention reduces computational costs by focusing on the most relevant parts of the input, while adaptive spans dynamically adjust the attention ranges. Bilinear attention complements both by modeling complex token interactions within this refined context. BiSparse-AAS consistently outperforms state-of-the-art baselines in both extractive and abstractive summarization tasks, achieving average ROUGE improvements of about 68.1% on CNN/DailyMail and 52.6% on XSum, while maintaining strong performance on OpenWebText and Gigaword datasets. By addressing efficiency, scalability, and long-sequence modeling, BiSparse-AAS provides a unified, practical solution for real-world text summarization applications. 

---
# Effect of Domain Generalization Techniques in Low Resource Systems 

**Authors**: Mahi Aminu, Chisom Chibuike, Fatimo Adebanjo, Omokolade Awosanya, Samuel Oyeneye  

**Link**: [PDF](https://arxiv.org/pdf/2510.27512)  

**Abstract**: Machine learning models typically assume that training and test data follow the same distribution, an assumption that often fails in real-world scenarios due to distribution shifts. This issue is especially pronounced in low-resource settings, where data scarcity and limited domain diversity hinder robust generalization. Domain generalization (DG) approaches address this challenge by learning features that remain invariant across domains, often using causal mechanisms to improve model robustness. In this study, we examine two distinct causal DG techniques in low-resource natural language tasks. First, we investigate a causal data augmentation (CDA) approach that automatically generates counterfactual examples to improve robustness to spurious correlations. We apply this method to sentiment classification on the NaijaSenti Twitter corpus, expanding the training data with semantically equivalent paraphrases to simulate controlled distribution shifts. Second, we explore an invariant causal representation learning (ICRL) approach using the DINER framework, originally proposed for debiasing aspect-based sentiment analysis. We adapt DINER to a multilingual setting. Our findings demonstrate that both approaches enhance robustness to unseen domains: counterfactual data augmentation yields consistent cross-domain accuracy gains in sentiment classification, while causal representation learning with DINER improves out-of-distribution performance in multilingual sentiment analysis, albeit with varying gains across languages. 

---
# The aftermath of compounds: Investigating Compounds and their Semantic Representations 

**Authors**: Swarang Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2510.27477)  

**Abstract**: This study investigates how well computational embeddings align with human semantic judgments in the processing of English compound words. We compare static word vectors (GloVe) and contextualized embeddings (BERT) against human ratings of lexeme meaning dominance (LMD) and semantic transparency (ST) drawn from a psycholinguistic dataset. Using measures of association strength (Edinburgh Associative Thesaurus), frequency (BNC), and predictability (LaDEC), we compute embedding-derived LMD and ST metrics and assess their relationships with human judgments via Spearmans correlation and regression analyses. Our results show that BERT embeddings better capture compositional semantics than GloVe, and that predictability ratings are strong predictors of semantic transparency in both human and model data. These findings advance computational psycholinguistics by clarifying the factors that drive compound word processing and offering insights into embedding-based semantic modeling. 

---
# Diffuse Thinking: Exploring Diffusion Language Models as Efficient Thought Proposers for Reasoning 

**Authors**: Chenyang Shao, Sijian Ren, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.27469)  

**Abstract**: In recent years, large language models (LLMs) have witnessed remarkable advancements, with the test-time scaling law consistently enhancing the reasoning capabilities. Through systematic evaluation and exploration of a diverse spectrum of intermediate thoughts, LLMs demonstrate the potential to generate deliberate reasoning steps, thereby substantially enhancing reasoning accuracy. However, LLMs' autoregressive generation paradigm results in reasoning performance scaling sub-optimally with test-time computation, often requiring excessive computational overhead to propose thoughts while yielding only marginal performance gains. In contrast, diffusion language models (DLMs) can efficiently produce diverse samples through parallel denoising in a single forward pass, inspiring us to leverage them for proposing intermediate thoughts, thereby alleviating the computational burden associated with autoregressive generation while maintaining quality. In this work, we propose an efficient collaborative reasoning framework, leveraging DLMs to generate candidate thoughts and LLMs to evaluate their quality. Experiments across diverse benchmarks demonstrate that our framework achieves strong performance in complex reasoning tasks, offering a promising direction for future research. Our code is open-source at this https URL. 

---
# VCORE: Variance-Controlled Optimization-based Reweighting for Chain-of-Thought Supervision 

**Authors**: Xuan Gong, Senmiao Wang, Hanbo Huang, Ruoyu Sun, Shiyu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27462)  

**Abstract**: Supervised fine-tuning (SFT) on long chain-of-thought (CoT) trajectories has emerged as a crucial technique for enhancing the reasoning abilities of large language models (LLMs). However, the standard cross-entropy loss treats all tokens equally, ignoring their heterogeneous contributions across a reasoning trajectory. This uniform treatment leads to misallocated supervision and weak generalization, especially in complex, long-form reasoning tasks. To address this, we introduce \textbf{V}ariance-\textbf{C}ontrolled \textbf{O}ptimization-based \textbf{RE}weighting (VCORE), a principled framework that reformulates CoT supervision as a constrained optimization problem. By adopting an optimization-theoretic perspective, VCORE enables a principled and adaptive allocation of supervision across tokens, thereby aligning the training objective more closely with the goal of robust reasoning generalization. Empirical evaluations demonstrate that VCORE consistently outperforms existing token reweighting methods. Across both in-domain and out-of-domain settings, VCORE achieves substantial performance gains on mathematical and coding benchmarks, using models from the Qwen3 series (4B, 8B, 32B) and LLaMA-3.1-8B-Instruct. Moreover, we show that VCORE serves as a more effective initialization for subsequent reinforcement learning, establishing a stronger foundation for advancing the reasoning capabilities of LLMs. The Code will be released at this https URL. 

---
# Dynamic Affective Memory Management for Personalized LLM Agents 

**Authors**: Junfeng Lu, Yueyan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.27418)  

**Abstract**: Advances in large language models are making personalized AI agents a new research focus. While current agent systems primarily rely on personalized external memory databases to deliver customized experiences, they face challenges such as memory redundancy, memory staleness, and poor memory-context integration, largely due to the lack of effective memory updates during interaction. To tackle these issues, we propose a new memory management system designed for affective scenarios. Our approach employs a Bayesian-inspired memory update algorithm with the concept of memory entropy, enabling the agent to autonomously maintain a dynamically updated memory vector database by minimizing global entropy to provide more personalized services. To better evaluate the system's effectiveness in this context, we propose DABench, a benchmark focusing on emotional expression and emotional change toward objects. Experimental results demonstrate that, our system achieves superior performance in personalization, logical coherence, and accuracy. Ablation studies further validate the effectiveness of the Bayesian-inspired update mechanism in alleviating memory bloat. Our work offers new insights into the design of long-term memory systems. 

---
# Awal -- Community-Powered Language Technology for Tamazight 

**Authors**: Alp Öktem, Farida Boudichat  

**Link**: [PDF](https://arxiv.org/pdf/2510.27407)  

**Abstract**: This paper presents Awal, a community-powered initiative for developing language technology resources for Tamazight. We provide a comprehensive review of the NLP landscape for Tamazight, examining recent progress in computational resources, and the emergence of community-driven approaches to address persistent data scarcity. Launched in 2024, this http URL platform addresses the underrepresentation of Tamazight in digital spaces through a collaborative platform enabling speakers to contribute translation and voice data. We analyze 18 months of community engagement, revealing significant barriers to participation including limited confidence in written Tamazight and ongoing standardization challenges. Despite widespread positive reception, actual data contribution remained concentrated among linguists and activists. The modest scale of community contributions -- 6,421 translation pairs and 3 hours of speech data -- highlights the limitations of applying standard crowdsourcing approaches to languages with complex sociolinguistic contexts. We are working on improved open-source MT models using the collected data. 

---
# Balancing Knowledge Updates: Toward Unified Modular Editing in LLMs 

**Authors**: Jiahao Liu, Zijian Wang, Kuo Zhao, Dong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27400)  

**Abstract**: Knowledge editing has emerged as an efficient approach for updating factual knowledge in large language models (LLMs). It typically locates knowledge storage modules and then modifies their parameters. However, most existing methods focus on the weights of multilayer perceptron (MLP) modules, which are often identified as the main repositories of factual information. Other components, such as attention (Attn) modules, are often ignored during editing. This imbalance can leave residual outdated knowledge and limit editing effectiveness. We perform comprehensive knowledge localization experiments on advanced LLMs and find that Attn modules play a substantial role in factual knowledge storage and retrieval, especially in earlier layers. Based on these insights, we propose IntAttn-Edit, a method that extends the associative memory paradigm to jointly update both MLP and Attn modules. Our approach uses a knowledge balancing strategy that allocates update magnitudes in proportion to each module's measured contribution to knowledge storage. Experiments on standard benchmarks show that IntAttn-Edit achieves higher edit success, better generalization, and stronger knowledge preservation than prior methods. Further analysis shows that the balancing strategy keeps editing performance within an optimal range across diverse settings. 

---
# From the Rock Floor to the Cloud: A Systematic Survey of State-of-the-Art NLP in Battery Life Cycle 

**Authors**: Tosin Adewumi, Martin Karlsson, Marcus Liwicki, Mikael Sjödahl, Lama Alkhaled, Rihab Gargouri, Nudrat Habib, Franz Hennie  

**Link**: [PDF](https://arxiv.org/pdf/2510.27369)  

**Abstract**: We present a comprehensive systematic survey of the application of natural language processing (NLP) along the entire battery life cycle, instead of one stage or method, and introduce a novel technical language processing (TLP) framework for the EU's proposed digital battery passport (DBP) and other general battery predictions. We follow the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) method and employ three reputable databases or search engines, including Google Scholar, Institute of Electrical and Electronics Engineers Xplore (IEEE Xplore), and Scopus. Consequently, we assessed 274 scientific papers before the critical review of the final 66 relevant papers. We publicly provide artifacts of the review for validation and reproducibility. The findings show that new NLP tasks are emerging in the battery domain, which facilitate materials discovery and other stages of the life cycle. Notwithstanding, challenges remain, such as the lack of standard benchmarks. Our proposed TLP framework, which incorporates agentic AI and optimized prompts, will be apt for tackling some of the challenges. 

---
# ThoughtProbe: Classifier-Guided LLM Thought Space Exploration via Probing Representations 

**Authors**: Zijian Wang, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27355)  

**Abstract**: This paper introduces ThoughtProbe, a novel inference time framework that leverages the hidden reasoning features of Large Language Models (LLMs) to improve their reasoning performance. Unlike previous works that manipulate the hidden representations to steer LLM generation, we harness them as discriminative signals to guide the tree structured response space exploration. In each node expansion, a classifier serves as a scoring and ranking mechanism that efficiently allocates computational resources by prioritizing higher score candidates for continuation. After completing the tree expansion, we collect answers from all branches to form a candidate answer pool. We then propose a branch aggregation method that marginalizes over all supporting branches by aggregating their CoT scores, thereby identifying the optimal answer from the pool. Experimental results show that our framework's comprehensive exploration not only covers valid reasoning chains but also effectively identifies them, achieving significant improvements across multiple arithmetic reasoning benchmarks. 

---
# TransAlign: Machine Translation Encoders are Strong Word Aligners, Too 

**Authors**: Benedikt Ebing, Christian Goldschmied, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2510.27337)  

**Abstract**: In the absence of sizable training data for most world languages and NLP tasks, translation-based strategies such as translate-test -- evaluating on noisy source language data translated from the target language -- and translate-train -- training on noisy target language data translated from the source language -- have been established as competitive approaches for cross-lingual transfer (XLT). For token classification tasks, these strategies require label projection: mapping the labels from each token in the original sentence to its counterpart(s) in the translation. To this end, it is common to leverage multilingual word aligners (WAs) derived from encoder language models such as mBERT or LaBSE. Despite obvious associations between machine translation (MT) and WA, research on extracting alignments with MT models is largely limited to exploiting cross-attention in encoder-decoder architectures, yielding poor WA results. In this work, in contrast, we propose TransAlign, a novel word aligner that utilizes the encoder of a massively multilingual MT model. We show that TransAlign not only achieves strong WA performance but substantially outperforms popular WA and state-of-the-art non-WA-based label projection methods in MT-based XLT for token classification. 

---
# A Unified Representation Underlying the Judgment of Large Language Models 

**Authors**: Yi-Long Lu, Jiajun Song, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27328)  

**Abstract**: A central architectural question for both biological and artificial intelligence is whether judgment relies on specialized modules or a unified, domain-general resource. While the discovery of decodable neural representations for distinct concepts in Large Language Models (LLMs) has suggested a modular architecture, whether these representations are truly independent systems remains an open question. Here we provide evidence for a convergent architecture. Across a range of LLMs, we find that diverse evaluative judgments are computed along a dominant dimension, which we term the Valence-Assent Axis (VAA). This axis jointly encodes subjective valence ("what is good") and the model's assent to factual claims ("what is true"). Through direct interventions, we show this unified representation creates a critical dependency: the VAA functions as a control signal that steers the generative process to construct a rationale consistent with its evaluative state, even at the cost of factual accuracy. This mechanism, which we term the subordination of reasoning, shifts the process of reasoning from impartial inference toward goal-directed justification. Our discovery offers a mechanistic account for systemic bias and hallucination, revealing how an architecture that promotes coherent judgment can systematically undermine faithful reasoning. 

---
# Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models? 

**Authors**: Deokhyung Kang, Seonjeong Hwang, Daehui Kim, Hyounghun Kim, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.27269)  

**Abstract**: Reasoning language models (RLMs) achieve strong performance on complex reasoning tasks, yet they still suffer from a multilingual reasoning gap, performing better in high-resource languages than in low-resource ones. While recent efforts have reduced this gap, its underlying causes remain largely unexplored. In this paper, we address this by showing that the multilingual reasoning gap largely stems from failures in language understanding-the model's inability to represent the multilingual input meaning into the dominant language (i.e., English) within its reasoning trace. This motivates us to examine whether understanding failures can be detected, as this ability could help mitigate the multilingual reasoning gap. To this end, we evaluate a range of detection methods and find that understanding failures can indeed be identified, with supervised approaches performing best. Building on this, we propose Selective Translation, a simple yet effective strategy that translates the multilingual input into English only when an understanding failure is detected. Experimental results show that Selective Translation bridges the multilingual reasoning gap, achieving near full-translation performance while using translation for only about 20% of inputs. Together, our work demonstrates that understanding failures are the primary cause of the multilingual reasoning gap and can be detected and selectively mitigated, providing key insight into its origin and a promising path toward more equitable multilingual reasoning. Our code and data are publicly available at this https URL. 

---
# MedCalc-Eval and MedCalc-Env: Advancing Medical Calculation Capabilities of Large Language Models 

**Authors**: Kangkun Mao, Jinru Ding, Jiayuan Chen, Mouxiao Bian, Ruiyao Chen, Xinwei Peng, Sijie Ren, Linyang Li, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27267)  

**Abstract**: As large language models (LLMs) enter the medical domain, most benchmarks evaluate them on question answering or descriptive reasoning, overlooking quantitative reasoning critical to clinical decision-making. Existing datasets like MedCalc-Bench cover few calculation tasks and fail to reflect real-world computational scenarios.
We introduce MedCalc-Eval, the largest benchmark for assessing LLMs' medical calculation abilities, comprising 700+ tasks across two types: equation-based (e.g., Cockcroft-Gault, BMI, BSA) and rule-based scoring systems (e.g., Apgar, Glasgow Coma Scale). These tasks span diverse specialties including internal medicine, surgery, pediatrics, and cardiology, offering a broader and more challenging evaluation setting.
To improve performance, we further develop MedCalc-Env, a reinforcement learning environment built on the InternBootcamp framework, enabling multi-step clinical reasoning and planning. Fine-tuning a Qwen2.5-32B model within this environment achieves state-of-the-art results on MedCalc-Eval, with notable gains in numerical sensitivity, formula selection, and reasoning robustness. Remaining challenges include unit conversion, multi-condition logic, and contextual understanding.
Code and datasets are available at this https URL. 

---
# Languages are Modalities: Cross-Lingual Alignment via Encoder Injection 

**Authors**: Rajan Agarwal, Aarush Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.27254)  

**Abstract**: Instruction-tuned Large Language Models (LLMs) underperform on low resource, non-Latin scripts due to tokenizer fragmentation and weak cross-lingual coupling. We present LLINK (Latent Language Injection for Non-English Knowledge), a compute efficient language-as-modality method that conditions an instruction-tuned decoder without changing the tokenizer or retraining the decoder. First, we align sentence embeddings from a frozen multilingual encoder to the decoder's latent embedding space at a reserved position via a lightweight contrastive projector. Second, the vector is expanded into K soft slots and trained with minimal adapters so the frozen decoder consumes the signal. LLINK substantially improves bilingual retrieval and achieves 81.3% preference over the base model and 63.6% over direct fine-tuning in LLM-judged Q&A evaluations. We further find that improvements can be attributed to reduced tokenization inflation and a stronger cross lingual alignment, despite the model having residual weaknesses in numeric fidelity. Treating low resource languages as a modality offers a practical path to stronger cross-lingual alignment in lightweight LLMs. 

---
# Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs 

**Authors**: Mohammad Tavakoli, Alireza Salemi, Carrie Ye, Mohamed Abdalla, Hamed Zamani, J Ross Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2510.27246)  

**Abstract**: Evaluating the abilities of large language models (LLMs) for tasks that require long-term memory and thus long-context reasoning, for example in conversational settings, is hampered by the existing benchmarks, which often lack narrative coherence, cover narrow domains, and only test simple recall-oriented tasks. This paper introduces a comprehensive solution to these challenges. First, we present a novel framework for automatically generating long (up to 10M tokens), coherent, and topically diverse conversations, accompanied by probing questions targeting a wide range of memory abilities. From this, we construct BEAM, a new benchmark comprising 100 conversations and 2,000 validated questions. Second, to enhance model performance, we propose LIGHT-a framework inspired by human cognition that equips LLMs with three complementary memory systems: a long-term episodic memory, a short-term working memory, and a scratchpad for accumulating salient facts. Our experiments on BEAM reveal that even LLMs with 1M token context windows (with and without retrieval-augmentation) struggle as dialogues lengthen. In contrast, LIGHT consistently improves performance across various models, achieving an average improvement of 3.5%-12.69% over the strongest baselines, depending on the backbone LLM. An ablation study further confirms the contribution of each memory component. 

---
# Identifying the Periodicity of Information in Natural Language 

**Authors**: Yulin Ou, Yu Wang, Yang Xu, Hendrik Buschmeier  

**Link**: [PDF](https://arxiv.org/pdf/2510.27241)  

**Abstract**: Recent theoretical advancement of information density in natural language has brought the following question on desk: To what degree does natural language exhibit periodicity pattern in its encoded information? We address this question by introducing a new method called AutoPeriod of Surprisal (APS). APS adopts a canonical periodicity detection algorithm and is able to identify any significant periods that exist in the surprisal sequence of a single document. By applying the algorithm to a set of corpora, we have obtained the following interesting results: Firstly, a considerable proportion of human language demonstrates a strong pattern of periodicity in information; Secondly, new periods that are outside the distributions of typical structural units in text (e.g., sentence boundaries, elementary discourse units, etc.) are found and further confirmed via harmonic regression modeling. We conclude that the periodicity of information in language is a joint outcome from both structured factors and other driving factors that take effect at longer distances. The advantages of our periodicity detection method and its potentials in LLM-generation detection are further discussed. 

---
# MemeArena: Automating Context-Aware Unbiased Evaluation of Harmfulness Understanding for Multimodal Large Language Models 

**Authors**: Zixin Chen, Hongzhan Lin, Kaixin Li, Ziyang Luo, Yayue Deng, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.27196)  

**Abstract**: The proliferation of memes on social media necessitates the capabilities of multimodal Large Language Models (mLLMs) to effectively understand multimodal harmfulness. Existing evaluation approaches predominantly focus on mLLMs' detection accuracy for binary classification tasks, which often fail to reflect the in-depth interpretive nuance of harmfulness across diverse contexts. In this paper, we propose MemeArena, an agent-based arena-style evaluation framework that provides a context-aware and unbiased assessment for mLLMs' understanding of multimodal harmfulness. Specifically, MemeArena simulates diverse interpretive contexts to formulate evaluation tasks that elicit perspective-specific analyses from mLLMs. By integrating varied viewpoints and reaching consensus among evaluators, it enables fair and unbiased comparisons of mLLMs' abilities to interpret multimodal harmfulness. Extensive experiments demonstrate that our framework effectively reduces the evaluation biases of judge agents, with judgment results closely aligning with human preferences, offering valuable insights into reliable and comprehensive mLLM evaluations in multimodal harmfulness understanding. Our code and data are publicly available at this https URL. 

---
# Simple Additions, Substantial Gains: Expanding Scripts, Languages, and Lineage Coverage in URIEL+ 

**Authors**: Mason Shipton, York Hay Ng, Aditya Khan, Phuong Hanh Hoang, Xiang Lu, A. Seza Doğruöz, En-Shiun Annie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.27183)  

**Abstract**: The URIEL+ linguistic knowledge base supports multilingual research by encoding languages through geographic, genetic, and typological vectors. However, data sparsity remains prevalent, in the form of missing feature types, incomplete language entries, and limited genealogical coverage. This limits the usefulness of URIEL+ in cross-lingual transfer, particularly for supporting low-resource languages. To address this sparsity, this paper extends URIEL+ with three contributions: introducing script vectors to represent writing system properties for 7,488 languages, integrating Glottolog to add 18,710 additional languages, and expanding lineage imputation for 26,449 languages by propagating typological and script features across genealogies. These additions reduce feature sparsity by 14% for script vectors, increase language coverage by up to 19,015 languages (1,007%), and improve imputation quality metrics by up to 33%. Our benchmark on cross-lingual transfer tasks (oriented around low-resource languages) shows occasionally divergent performance compared to URIEL+, with performance gains up to 6% in certain setups. Our advances make URIEL+ more complete and inclusive for multilingual research. 

---
# Probability Distributions Computed by Hard-Attention Transformers 

**Authors**: Andy Yang, Anej Svete, Jiaoda Li, Anthony Widjaja Lin, Jonathan Rawski, Ryan Cotterell, David Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27118)  

**Abstract**: Most expressivity results for transformers treat them as language recognizers (which accept or reject strings), and not as they are used in practice, as language models (which generate strings autoregressively and probabilistically). Here, we characterize the probability distributions that transformer language models can express. We show that making transformer language recognizers autoregressive can sometimes increase their expressivity, and that making them probabilistic can break equivalences that hold in the non-probabilistic case. Our overall contribution is to tease apart what functions transformers are capable of expressing, in their most common use-case as language models. 

---
# Rating Roulette: Self-Inconsistency in LLM-As-A-Judge Frameworks 

**Authors**: Rajarshi Haldar, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2510.27106)  

**Abstract**: As Natural Language Generation (NLG) continues to be widely adopted, properly assessing it has become quite difficult. Lately, using large language models (LLMs) for evaluating these generations has gained traction, as they tend to align more closely with human preferences than conventional n-gram or embedding-based metrics. In our experiments, we show that LLM judges have low intra-rater reliability in their assigned scores across different runs. This variance makes their ratings inconsistent, almost arbitrary in the worst case, making it difficult to measure how good their judgments actually are. We quantify this inconsistency across different NLG tasks and benchmarks and see if judicious use of LLM judges can still be useful following proper guidelines. 

---
# Characterizing Selective Refusal Bias in Large Language Models 

**Authors**: Adel Khorramrouz, Sharon Levy  

**Link**: [PDF](https://arxiv.org/pdf/2510.27087)  

**Abstract**: Safety guardrails in large language models(LLMs) are developed to prevent malicious users from generating toxic content at a large scale. However, these measures can inadvertently introduce or reflect new biases, as LLMs may refuse to generate harmful content targeting some demographic groups and not others. We explore this selective refusal bias in LLM guardrails through the lens of refusal rates of targeted individual and intersectional demographic groups, types of LLM responses, and length of generated refusals. Our results show evidence of selective refusal bias across gender, sexual orientation, nationality, and religion attributes. This leads us to investigate additional safety implications via an indirect attack, where we target previously refused groups. Our findings emphasize the need for more equitable and robust performance in safety guardrails across demographic groups. 

---
# Contrastive Knowledge Transfer and Robust Optimization for Secure Alignment of Large Language Models 

**Authors**: Jiasen Zheng, Huajun Zhang, Xu Yan, Ran Hao, Chong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.27077)  

**Abstract**: This paper addresses the limitations of large-scale language models in safety alignment and robustness by proposing a fine-tuning method that combines contrastive distillation with noise-robust training. The method freezes the backbone model and transfers the knowledge boundaries of the teacher model to the student model through distillation, thereby improving semantic consistency and alignment accuracy. At the same time, noise perturbations and robust optimization constraints are introduced during training to ensure that the model maintains stable predictive outputs under noisy and uncertain inputs. The overall framework consists of distillation loss, robustness loss, and a regularization term, forming a unified optimization objective that balances alignment ability with resistance to interference. To systematically validate its effectiveness, the study designs experiments from multiple perspectives, including distillation weight sensitivity, stability analysis under computation budgets and mixed-precision environments, and the impact of data noise and distribution shifts on model performance. Results show that the method significantly outperforms existing baselines in knowledge transfer, robustness, and overall safety, achieving the best performance across several key metrics. This work not only enriches the theoretical system of parameter-efficient fine-tuning but also provides a new solution for building safer and more trustworthy alignment mechanisms. 

---
# Detecting Data Contamination in LLMs via In-Context Learning 

**Authors**: Michał Zawalski, Meriem Boubdir, Klaudia Bałazy, Besmira Nushi, Pablo Ribalta  

**Link**: [PDF](https://arxiv.org/pdf/2510.27055)  

**Abstract**: We present Contamination Detection via Context (CoDeC), a practical and accurate method to detect and quantify training data contamination in large language models. CoDeC distinguishes between data memorized during training and data outside the training distribution by measuring how in-context learning affects model performance. We find that in-context examples typically boost confidence for unseen datasets but may reduce it when the dataset was part of training, due to disrupted memorization patterns. Experiments show that CoDeC produces interpretable contamination scores that clearly separate seen and unseen datasets, and reveals strong evidence of memorization in open-weight models with undisclosed training corpora. The method is simple, automated, and both model- and dataset-agnostic, making it easy to integrate with benchmark evaluations. 

---
# LLM-Centric RAG with Multi-Granular Indexing and Confidence Constraints 

**Authors**: Xiaofan Guo, Yaxuan Luan, Yue Kang, Xiangchen Song, Jinxu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.27054)  

**Abstract**: This paper addresses the issues of insufficient coverage, unstable results, and limited reliability in retrieval-augmented generation under complex knowledge environments, and proposes a confidence control method that integrates multi-granularity memory indexing with uncertainty estimation. The method builds a hierarchical memory structure that divides knowledge representations into different levels of granularity, enabling dynamic indexing and retrieval from local details to global context, and thus establishing closer semantic connections between retrieval and generation. On this basis, an uncertainty estimation mechanism is introduced to explicitly constrain and filter low-confidence paths during the generation process, allowing the model to maintain information coverage while effectively suppressing noise and false content. The overall optimization objective consists of generation loss, entropy constraints, and variance regularization, forming a unified confidence control framework. In the experiments, comprehensive sensitivity tests and comparative analyses were designed, covering hyperparameters, environmental conditions, and data structures, to verify the stability and robustness of the proposed method across different scenarios. The results show that the method achieves superior performance over existing models in QA accuracy, retrieval recall, ranking quality, and factual consistency, demonstrating the effectiveness of combining multi-granularity indexing with confidence control. This study not only provides a new technical pathway for retrieval-augmented generation but also offers practical evidence for improving the reliability and controllability of large models in complex contexts. 

---
# VISTA Score: Verification In Sequential Turn-based Assessment 

**Authors**: Ashley Lewis, Andrew Perrault, Eric Fosler-Lussier, Michael White  

**Link**: [PDF](https://arxiv.org/pdf/2510.27052)  

**Abstract**: Hallucination--defined here as generating statements unsupported or contradicted by available evidence or conversational context--remains a major obstacle to deploying conversational AI systems in settings that demand factual reliability. Existing metrics either evaluate isolated responses or treat unverifiable content as errors, limiting their use for multi-turn dialogue. We introduce VISTA (Verification In Sequential Turn-based Assessment), a framework for evaluating conversational factuality through claim-level verification and sequential consistency tracking. VISTA decomposes each assistant turn into atomic factual claims, verifies them against trusted sources and dialogue history, and categorizes unverifiable statements (subjective, contradicted, lacking evidence, or abstaining). Across eight large language models and four dialogue factuality benchmarks (AIS, BEGIN, FAITHDIAL, and FADE), VISTA substantially improves hallucination detection over FACTSCORE and LLM-as-Judge baselines. Human evaluation confirms that VISTA's decomposition improves annotator agreement and reveals inconsistencies in existing benchmarks. By modeling factuality as a dynamic property of conversation, VISTA offers a more transparent, human-aligned measure of truthfulness in dialogue systems. 

---
# Recursive numeral systems are highly regular and easy to process 

**Authors**: Ponrawee Prasertsom, Andrea Silvi, Jennifer Culbertson, Moa Johansson, Devdatt Dubhashi, Kenny Smith  

**Link**: [PDF](https://arxiv.org/pdf/2510.27049)  

**Abstract**: Previous work has argued that recursive numeral systems optimise the trade-off between lexicon size and average morphosyntatic complexity (Denić and Szymanik, 2024). However, showing that only natural-language-like systems optimise this tradeoff has proven elusive, and the existing solution has relied on ad-hoc constraints to rule out unnatural systems (Yang and Regier, 2025). Here, we argue that this issue arises because the proposed trade-off has neglected regularity, a crucial aspect of complexity central to human grammars in general. Drawing on the Minimum Description Length (MDL) approach, we propose that recursive numeral systems are better viewed as efficient with regard to their regularity and processing complexity. We show that our MDL-based measures of regularity and processing complexity better capture the key differences between attested, natural systems and unattested but possible ones, including "optimal" recursive numeral systems from previous work, and that the ad-hoc constraints from previous literature naturally follow from regularity. Our approach highlights the need to incorporate regularity across sets of forms in studies that attempt to measure and explain optimality in language. 

---
# Quantitative Intertextuality from the Digital Humanities Perspective: A Survey 

**Authors**: Siyu Duan  

**Link**: [PDF](https://arxiv.org/pdf/2510.27045)  

**Abstract**: The connection between texts is referred to as intertextuality in literary theory, which served as an important theoretical basis in many digital humanities studies. Over the past decade, advancements in natural language processing have ushered intertextuality studies into the quantitative age. Large-scale intertextuality research based on cutting-edge methods has continuously emerged. This paper provides a roadmap for quantitative intertextuality studies, summarizing their data, methods, and applications. Drawing on data from multiple languages and topics, this survey reviews methods from statistics to deep learning. It also summarizes their applications in humanities and social sciences research and the associated platform tools. Driven by advances in computer technology, more precise, diverse, and large-scale intertext studies can be anticipated. Intertextuality holds promise for broader application in interdisciplinary research bridging AI and the humanities. 

---
# Dataset Creation and Baseline Models for Sexism Detection in Hausa 

**Authors**: Fatima Adam Muhammad, Shamsuddeen Muhammad Hassan, Isa Inuwa-Dutse  

**Link**: [PDF](https://arxiv.org/pdf/2510.27038)  

**Abstract**: Sexism reinforces gender inequality and social exclusion by perpetuating stereotypes, bias, and discriminatory norms. Noting how online platforms enable various forms of sexism to thrive, there is a growing need for effective sexism detection and mitigation strategies. While computational approaches to sexism detection are widespread in high-resource languages, progress remains limited in low-resource languages where limited linguistic resources and cultural differences affect how sexism is expressed and perceived. This study introduces the first Hausa sexism detection dataset, developed through community engagement, qualitative coding, and data augmentation. For cultural nuances and linguistic representation, we conducted a two-stage user study (n=66) involving native speakers to explore how sexism is defined and articulated in everyday discourse. We further experiment with both traditional machine learning classifiers and pre-trained multilingual language models and evaluating the effectiveness few-shot learning in detecting sexism in Hausa. Our findings highlight challenges in capturing cultural nuance, particularly with clarification-seeking and idiomatic expressions, and reveal a tendency for many false positives in such cases. 

---
# Elastic Architecture Search for Efficient Language Models 

**Authors**: Shang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27037)  

**Abstract**: As large pre-trained language models become increasingly critical to natural language understanding (NLU) tasks, their substantial computational and memory requirements have raised significant economic and environmental concerns. Addressing these challenges, this paper introduces the Elastic Language Model (ELM), a novel neural architecture search (NAS) method optimized for compact language models. ELM extends existing NAS approaches by introducing a flexible search space with efficient transformer blocks and dynamic modules for dimension and head number adjustment. These innovations enhance the efficiency and flexibility of the search process, which facilitates more thorough and effective exploration of model architectures. We also introduce novel knowledge distillation losses that preserve the unique characteristics of each block, in order to improve the discrimination between architectural choices during the search process. Experiments on masked language modeling and causal language modeling tasks demonstrate that models discovered by ELM significantly outperform existing methods. 

---
# Kad: A Framework for Proxy-based Test-time Alignment with Knapsack Approximation Deferral 

**Authors**: Ayoub Hammal, Pierre Zweigenbaum, Caio Corro  

**Link**: [PDF](https://arxiv.org/pdf/2510.27017)  

**Abstract**: Several previous works concluded that the largest part of generation capabilities of large language models (LLM) are learned (early) during pre-training. However, LLMs still require further alignment to adhere to downstream task requirements and stylistic preferences, among other desired properties. As LLMs continue to scale in terms of size, the computational cost of alignment procedures increase prohibitively. In this work, we propose a novel approach to circumvent these costs via proxy-based test-time alignment, i.e. using guidance from a small aligned model. Our approach can be described as token-specific cascading method, where the token-specific deferral rule is reduced to 0-1 knapsack problem. In this setting, we derive primal and dual approximations of the optimal deferral decision. We experimentally show the benefits of our method both in task performance and speculative decoding speed. 

---
# Semantically-Aware LLM Agent to Enhance Privacy in Conversational AI Services 

**Authors**: Jayden Serenari, Stephen Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.27016)  

**Abstract**: With the increasing use of conversational AI systems, there is growing concern over privacy leaks, especially when users share sensitive personal data in interactions with Large Language Models (LLMs). Conversations shared with these models may contain Personally Identifiable Information (PII), which, if exposed, could lead to security breaches or identity theft. To address this challenge, we present the Local Optimizations for Pseudonymization with Semantic Integrity Directed Entity Detection (LOPSIDED) framework, a semantically-aware privacy agent designed to safeguard sensitive PII data when using remote LLMs. Unlike prior work that often degrade response quality, our approach dynamically replaces sensitive PII entities in user prompts with semantically consistent pseudonyms, preserving the contextual integrity of conversations. Once the model generates its response, the pseudonyms are automatically depseudonymized, ensuring the user receives an accurate, privacy-preserving output. We evaluate our approach using real-world conversations sourced from ShareGPT, which we further augment and annotate to assess whether named entities are contextually relevant to the model's response. Our results show that LOPSIDED reduces semantic utility errors by a factor of 5 compared to baseline techniques, all while enhancing privacy. 

---
# Overview of the MEDIQA-OE 2025 Shared Task on Medical Order Extraction from Doctor-Patient Consultations 

**Authors**: Jean-Philippe Corbeil, Asma Ben Abacha, Jerome Tremblay, Phillip Swazinna, Akila Jeeson Daniel, Miguel Del-Agua, Francois Beaulieu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26974)  

**Abstract**: Clinical documentation increasingly uses automatic speech recognition and summarization, yet converting conversations into actionable medical orders for Electronic Health Records remains unexplored. A solution to this problem can significantly reduce the documentation burden of clinicians and directly impact downstream patient care. We introduce the MEDIQA-OE 2025 shared task, the first challenge on extracting medical orders from doctor-patient conversations. Six teams participated in the shared task and experimented with a broad range of approaches, and both closed- and open-weight large language models (LLMs). In this paper, we describe the MEDIQA-OE task, dataset, final leaderboard ranking, and participants' solutions. 

---
# Frame Semantic Patterns for Identifying Underreporting of Notifiable Events in Healthcare: The Case of Gender-Based Violence 

**Authors**: Lívia Dutra, Arthur Lorenzi, Laís Berno, Franciany Campos, Karoline Biscardi, Kenneth Brown, Marcelo Viridiano, Frederico Belcavello, Ely Matos, Olívia Guaranha, Erik Santos, Sofia Reinach, Tiago Timponi Torrent  

**Link**: [PDF](https://arxiv.org/pdf/2510.26969)  

**Abstract**: We introduce a methodology for the identification of notifiable events in the domain of healthcare. The methodology harnesses semantic frames to define fine-grained patterns and search them in unstructured data, namely, open-text fields in e-medical records. We apply the methodology to the problem of underreporting of gender-based violence (GBV) in e-medical records produced during patients' visits to primary care units. A total of eight patterns are defined and searched on a corpus of 21 million sentences in Brazilian Portuguese extracted from e-SUS APS. The results are manually evaluated by linguists and the precision of each pattern measured. Our findings reveal that the methodology effectively identifies reports of violence with a precision of 0.726, confirming its robustness. Designed as a transparent, efficient, low-carbon, and language-agnostic pipeline, the approach can be easily adapted to other health surveillance contexts, contributing to the broader, ethical, and explainable use of NLP in public health systems. 

---
# Understanding and Enhancing Mamba-Transformer Hybrids for Memory Recall and Language Modeling 

**Authors**: Hyunji Lee, Wenhao Yu, Hongming Zhang, Kaixin Ma, Jiyeon Kim, Dong Yu, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2510.26912)  

**Abstract**: Hybrid models that combine state space models (SSMs) with attention mechanisms have shown strong performance by leveraging the efficiency of SSMs and the high recall ability of attention. However, the architectural design choices behind these hybrid models remain insufficiently understood. In this work, we analyze hybrid architectures through the lens of memory utilization and overall performance, and propose a complementary method to further enhance their effectiveness. We first examine the distinction between sequential and parallel integration of SSM and attention layers. Our analysis reveals several interesting findings, including that sequential hybrids perform better on shorter contexts, whereas parallel hybrids are more effective for longer contexts. We also introduce a data-centric approach of continually training on datasets augmented with paraphrases, which further enhances recall while preserving other capabilities. It generalizes well across different base models and outperforms architectural modifications aimed at enhancing recall. Our findings provide a deeper understanding of hybrid SSM-attention models and offer practical guidance for designing architectures tailored to various use cases. Our findings provide a deeper understanding of hybrid SSM-attention models and offer practical guidance for designing architectures tailored to various use cases. 

---
# Visual Backdoor Attacks on MLLM Embodied Decision Making via Contrastive Trigger Learning 

**Authors**: Qiusi Zhan, Hyeonjeong Ha, Rui Yang, Sirui Xu, Hanyang Chen, Liang-Yan Gui, Yu-Xiong Wang, Huan Zhang, Heng Ji, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27623)  

**Abstract**: Multimodal large language models (MLLMs) have advanced embodied agents by enabling direct perception, reasoning, and planning task-oriented actions from visual inputs. However, such vision driven embodied agents open a new attack surface: visual backdoor attacks, where the agent behaves normally until a visual trigger appears in the scene, then persistently executes an attacker-specified multi-step policy. We introduce BEAT, the first framework to inject such visual backdoors into MLLM-based embodied agents using objects in the environments as triggers. Unlike textual triggers, object triggers exhibit wide variation across viewpoints and lighting, making them difficult to implant reliably. BEAT addresses this challenge by (1) constructing a training set that spans diverse scenes, tasks, and trigger placements to expose agents to trigger variability, and (2) introducing a two-stage training scheme that first applies supervised fine-tuning (SFT) and then our novel Contrastive Trigger Learning (CTL). CTL formulates trigger discrimination as preference learning between trigger-present and trigger-free inputs, explicitly sharpening the decision boundaries to ensure precise backdoor activation. Across various embodied agent benchmarks and MLLMs, BEAT achieves attack success rates up to 80%, while maintaining strong benign task performance, and generalizes reliably to out-of-distribution trigger placements. Notably, compared to naive SFT, CTL boosts backdoor activation accuracy up to 39% under limited backdoor data. These findings expose a critical yet unexplored security risk in MLLM-based embodied agents, underscoring the need for robust defenses before real-world deployment. 

---
# Towards Universal Video Retrieval: Generalizing Video Embedding via Synthesized Multimodal Pyramid Curriculum 

**Authors**: Zhuoning Guo, Mingxin Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27571)  

**Abstract**: The prevailing video retrieval paradigm is structurally misaligned, as narrow benchmarks incentivize correspondingly limited data and single-task training. Therefore, universal capability is suppressed due to the absence of a diagnostic evaluation that defines and demands multi-dimensional generalization. To break this cycle, we introduce a framework built on the co-design of evaluation, data, and modeling. First, we establish the Universal Video Retrieval Benchmark (UVRB), a suite of 16 datasets designed not only to measure performance but also to diagnose critical capability gaps across tasks and domains. Second, guided by UVRB's diagnostics, we introduce a scalable synthesis workflow that generates 1.55 million high-quality pairs to populate the semantic space required for universality. Finally, we devise the Modality Pyramid, a curriculum that trains our General Video Embedder (GVE) by explicitly leveraging the latent interconnections within our diverse data. Extensive experiments show GVE achieves state-of-the-art zero-shot generalization on UVRB. In particular, our analysis reveals that popular benchmarks are poor predictors of general ability and that partially relevant retrieval is a dominant but overlooked scenario. Overall, our co-designed framework provides a practical path to escape the limited scope and advance toward truly universal video retrieval. 

---
# SIGMA: Search-Augmented On-Demand Knowledge Integration for Agentic Mathematical Reasoning 

**Authors**: Ali Asgarov, Umid Suleymanov, Aadyant Khatri  

**Link**: [PDF](https://arxiv.org/pdf/2510.27568)  

**Abstract**: Solving mathematical reasoning problems requires not only accurate access to relevant knowledge but also careful, multi-step thinking. However, current retrieval-augmented models often rely on a single perspective, follow inflexible search strategies, and struggle to effectively combine information from multiple sources. We introduce SIGMA (Search-Augmented On-Demand Knowledge Integration for AGentic Mathematical reAsoning), a unified framework that orchestrates specialized agents to independently reason, perform targeted searches, and synthesize findings through a moderator mechanism. Each agent generates hypothetical passages to optimize retrieval for its analytic perspective, ensuring knowledge integration is both context-sensitive and computation-efficient. When evaluated on challenging benchmarks such as MATH500, AIME, and PhD-level science QA GPQA, SIGMA consistently outperforms both open- and closed-source systems, achieving an absolute performance improvement of 7.4%. Our results demonstrate that multi-agent, on-demand knowledge integration significantly enhances both reasoning accuracy and efficiency, offering a scalable approach for complex, knowledge-intensive problem-solving. We will release the code upon publication. 

---
# Thought Branches: Interpreting LLM Reasoning Requires Resampling 

**Authors**: Uzay Macar, Paul C. Bogdan, Senthooran Rajamanoharan, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2510.27484)  

**Abstract**: Most work interpreting reasoning models studies only a single chain-of-thought (CoT), yet these models define distributions over many possible CoTs. We argue that studying a single sample is inadequate for understanding causal influence and the underlying computation. Though fully specifying this distribution is intractable, it can be understood by sampling. We present case studies using resampling to investigate model decisions. First, when a model states a reason for its action, does that reason actually cause the action? In "agentic misalignment" scenarios, we resample specific sentences to measure their downstream effects. Self-preservation sentences have small causal impact, suggesting they do not meaningfully drive blackmail. Second, are artificial edits to CoT sufficient for steering reasoning? These are common in literature, yet take the model off-policy. Resampling and selecting a completion with the desired property is a principled on-policy alternative. We find off-policy interventions yield small and unstable effects compared to resampling in decision-making tasks. Third, how do we understand the effect of removing a reasoning step when the model may repeat it post-edit? We introduce a resilience metric that repeatedly resamples to prevent similar content from reappearing downstream. Critical planning statements resist removal but have large effects when eliminated. Fourth, since CoT is sometimes "unfaithful", can our methods teach us anything in these settings? Adapting causal mediation analysis, we find that hints that have a causal effect on the output without being explicitly mentioned exert a subtle and cumulative influence on the CoT that persists even if the hint is removed. Overall, studying distributions via resampling enables reliable causal analysis, clearer narratives of model reasoning, and principled CoT interventions. 

---
# DeepCompress: A Dual Reward Strategy for Dynamically Exploring and Compressing Reasoning Chains 

**Authors**: Tian Liang, Wenxiang Jiao, Zhiwei He, Jiahao Xu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27419)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated impressive capabilities but suffer from cognitive inefficiencies like ``overthinking'' simple problems and ``underthinking'' complex ones. While existing methods that use supervised fine-tuning~(SFT) or reinforcement learning~(RL) with token-length rewards can improve efficiency, they often do so at the cost of accuracy. This paper introduces \textbf{DeepCompress}, a novel framework that simultaneously enhances both the accuracy and efficiency of LRMs. We challenge the prevailing approach of consistently favoring shorter reasoning paths, showing that longer responses can contain a broader range of correct solutions for difficult problems. DeepCompress employs an adaptive length reward mechanism that dynamically classifies problems as ``Simple'' or ``Hard'' in real-time based on the model's evolving capability. It encourages shorter, more efficient reasoning for ``Simple'' problems while promoting longer, more exploratory thought chains for ``Hard'' problems. This dual-reward strategy enables the model to autonomously adjust its Chain-of-Thought (CoT) length, compressing reasoning for well-mastered problems and extending it for those it finds challenging. Experimental results on challenging mathematical benchmarks show that DeepCompress consistently outperforms baseline methods, achieving superior accuracy while significantly improving token efficiency. 

---
# Atlas-Alignment: Making Interpretability Transferable Across Language Models 

**Authors**: Bruno Puri, Jim Berend, Sebastian Lapuschkin, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2510.27413)  

**Abstract**: Interpretability is crucial for building safe, reliable, and controllable language models, yet existing interpretability pipelines remain costly and difficult to scale. Interpreting a new model typically requires costly training of model-specific sparse autoencoders, manual or semi-automated labeling of SAE components, and their subsequent validation. We introduce Atlas-Alignment, a framework for transferring interpretability across language models by aligning unknown latent spaces to a Concept Atlas - a labeled, human-interpretable latent space - using only shared inputs and lightweight representational alignment techniques. Once aligned, this enables two key capabilities in previously opaque models: (1) semantic feature search and retrieval, and (2) steering generation along human-interpretable atlas concepts. Through quantitative and qualitative evaluations, we show that simple representational alignment methods enable robust semantic retrieval and steerable generation without the need for labeled concept data. Atlas-Alignment thus amortizes the cost of explainable AI and mechanistic interpretability: by investing in one high-quality Concept Atlas, we can make many new models transparent and controllable at minimal marginal cost. 

---
# Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity 

**Authors**: Austin Meek, Eitan Sprejer, Iván Arcuschin, Austin J. Brockmeier, Steven Basart  

**Link**: [PDF](https://arxiv.org/pdf/2510.27378)  

**Abstract**: Chain-of-thought (CoT) outputs let us read a model's step-by-step reasoning. Since any long, serial reasoning process must pass through this textual trace, the quality of the CoT is a direct window into what the model is thinking. This visibility could help us spot unsafe or misaligned behavior (monitorability), but only if the CoT is transparent about its internal reasoning (faithfulness). Fully measuring faithfulness is difficult, so researchers often focus on examining the CoT in cases where the model changes its answer after adding a cue to the input. This proxy finds some instances of unfaithfulness but loses information when the model maintains its answer, and does not investigate aspects of reasoning not tied to the cue. We extend these results to a more holistic sense of monitorability by introducing verbosity: whether the CoT lists every factor needed to solve the task. We combine faithfulness and verbosity into a single monitorability score that shows how well the CoT serves as the model's external `working memory', a property that many safety schemes based on CoT monitoring depend on. We evaluate instruction-tuned and reasoning models on BBH, GPQA, and MMLU. Our results show that models can appear faithful yet remain hard to monitor when they leave out key factors, and that monitorability differs sharply across model families. We release our evaluation code using the Inspect library to support reproducible future work. 

---
# Un-Attributability: Computing Novelty From Retrieval & Semantic Similarity 

**Authors**: Philipp Davydov, Ameya Prabhu, Matthias Bethge, Elisa Nguyen, Seong Joon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.27313)  

**Abstract**: Understanding how language-model outputs relate to the pretraining corpus is central to studying model behavior. Most training data attribution (TDA) methods ask which training examples causally influence a given output, often using leave-one-out tests. We invert the question: which outputs cannot be attributed to any pretraining example? We introduce un-attributability as an operational measure of semantic novelty: an output is novel if the pretraining corpus contains no semantically similar context. We approximate this with a simple two-stage retrieval pipeline: index the corpus with lightweight GIST embeddings, retrieve the top-n candidates, then rerank with ColBERTv2. If the nearest corpus item is less attributable than a human-generated text reference, we consider the output of the model as novel. We evaluate on SmolLM and SmolLM2 and report three findings: (1) models draw on pretraining data across much longer spans than previously reported; (2) some domains systematically promote or suppress novelty; and (3) instruction tuning not only alters style but also increases novelty. Reframing novelty assessment around un-attributability enables efficient analysis at pretraining scale. We release ~20 TB of corpus chunks and index artifacts to support replication and large-scale extension of our analysis at this https URL 

---
# Higher-order Linear Attention 

**Authors**: Yifan Zhang, Zhen Qin, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27258)  

**Abstract**: The quadratic cost of scaled dot-product attention is a central obstacle to scaling autoregressive language models to long contexts. Linear-time attention and State Space Models (SSMs) provide scalable alternatives but are typically restricted to first-order or kernel-based approximations, which can limit expressivity. We introduce Higher-order Linear Attention (HLA), a causal, streaming mechanism that realizes higher interactions via compact prefix sufficient statistics. In the second-order case, HLA maintains a constant-size state and computes per-token outputs in linear time without materializing any $n \times n$ matrices. We give closed-form streaming identities, a strictly causal masked variant using two additional summaries, and a chunk-parallel training scheme based on associative scans that reproduces the activations of a serial recurrence exactly. We further outline extensions to third and higher orders. Collectively, these results position HLA as a principled, scalable building block that combines attention-like, data-dependent mixing with the efficiency of modern recurrent architectures. Project Page: this https URL. 

---
# DRAMA: Unifying Data Retrieval and Analysis for Open-Domain Analytic Queries 

**Authors**: Chuxuan Hu, Maxwell Yang, James Weiland, Yeji Lim, Suhas Palawala, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27238)  

**Abstract**: Manually conducting real-world data analyses is labor-intensive and inefficient. Despite numerous attempts to automate data science workflows, none of the existing paradigms or systems fully demonstrate all three key capabilities required to support them effectively: (1) open-domain data collection, (2) structured data transformation, and (3) analytic reasoning.
To overcome these limitations, we propose DRAMA, an end-to-end paradigm that answers users' analytic queries in natural language on large-scale open-domain data. DRAMA unifies data collection, transformation, and analysis as a single pipeline. To quantitatively evaluate system performance on tasks representative of DRAMA, we construct a benchmark, DRAMA-Bench, consisting of two categories of tasks: claim verification and question answering, each comprising 100 instances. These tasks are derived from real-world applications that have gained significant public attention and require the retrieval and analysis of open-domain data. We develop DRAMA-Bot, a multi-agent system designed following DRAMA. It comprises a data retriever that collects and transforms data by coordinating the execution of sub-agents, and a data analyzer that performs structured reasoning over the retrieved data. We evaluate DRAMA-Bot on DRAMA-Bench together with five state-of-the-art baseline agents. DRAMA-Bot achieves 86.5% task accuracy at a cost of $0.05, outperforming all baselines with up to 6.9 times the accuracy and less than 1/6 of the cost. DRAMA is publicly available at this https URL. 

---
# Can MLLMs Read the Room? A Multimodal Benchmark for Verifying Truthfulness in Multi-Party Social Interactions 

**Authors**: Caixin Kang, Yifei Huang, Liangyang Ouyang, Mingfang Zhang, Yoichi Sato  

**Link**: [PDF](https://arxiv.org/pdf/2510.27195)  

**Abstract**: As AI systems become increasingly integrated into human lives, endowing them with robust social intelligence has emerged as a critical frontier. A key aspect of this intelligence is discerning truth from deception, a ubiquitous element of human interaction that is conveyed through a complex interplay of verbal language and non-verbal visual cues. However, automatic deception detection in dynamic, multi-party conversations remains a significant challenge. The recent rise of powerful Multimodal Large Language Models (MLLMs), with their impressive abilities in visual and textual understanding, makes them natural candidates for this task. Consequently, their capabilities in this crucial domain are mostly unquantified. To address this gap, we introduce a new task, Multimodal Interactive Veracity Assessment (MIVA), and present a novel multimodal dataset derived from the social deduction game Werewolf. This dataset provides synchronized video, text, with verifiable ground-truth labels for every statement. We establish a comprehensive benchmark evaluating state-of-the-art MLLMs, revealing a significant performance gap: even powerful models like GPT-4o struggle to distinguish truth from falsehood reliably. Our analysis of failure modes indicates that these models fail to ground language in visual social cues effectively and may be overly conservative in their alignment, highlighting the urgent need for novel approaches to building more perceptive and trustworthy AI systems. 

---
# Glia: A Human-Inspired AI for Automated Systems Design and Optimization 

**Authors**: Pouya Hamadanian, Pantea Karimi, Arash Nasr-Esfahany, Kimia Noorbakhsh, Joseph Chandler, Ali ParandehGheibi, Mohammad Alizadeh, Hari Balakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2510.27176)  

**Abstract**: Can an AI autonomously design mechanisms for computer systems on par with the creativity and reasoning of human experts? We present Glia, an AI architecture for networked systems design that uses large language models (LLMs) in a human-inspired, multi-agent workflow. Each agent specializes in reasoning, experimentation, and analysis, collaborating through an evaluation framework that grounds abstract reasoning in empirical feedback. Unlike prior ML-for-systems methods that optimize black-box policies, Glia generates interpretable designs and exposes its reasoning process. When applied to a distributed GPU cluster for LLM inference, it produces new algorithms for request routing, scheduling, and auto-scaling that perform at human-expert levels in significantly less time, while yielding novel insights into workload behavior. Our results suggest that by combining reasoning LLMs with structured experimentation, an AI can produce creative and understandable designs for complex systems problems. 

---
# Towards a Measure of Algorithm Similarity 

**Authors**: Shairoz Sohail, Taher Ali  

**Link**: [PDF](https://arxiv.org/pdf/2510.27063)  

**Abstract**: Given two algorithms for the same problem, can we determine whether they are meaningfully different? In full generality, the question is uncomputable, and empirically it is muddied by competing notions of similarity. Yet, in many applications (such as clone detection or program synthesis) a pragmatic and consistent similarity metric is necessary. We review existing equivalence and similarity notions and introduce EMOC: An Evaluation-Memory-Operations-Complexity framework that embeds algorithm implementations into a feature space suitable for downstream tasks. We compile PACD, a curated dataset of verified Python implementations across three problems, and show that EMOC features support clustering and classification of algorithm types, detection of near-duplicates, and quantification of diversity in LLM-generated programs. Code, data, and utilities for computing EMOC embeddings are released to facilitate reproducibility and future work on algorithm similarity. 

---
# Semantic Frame Aggregation-based Transformer for Live Video Comment Generation 

**Authors**: Anam Fatima, Yi Yu, Janak Kapuriya, Julien Lalanne, Jainendra Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2510.26978)  

**Abstract**: Live commenting on video streams has surged in popularity on platforms like Twitch, enhancing viewer engagement through dynamic interactions. However, automatically generating contextually appropriate comments remains a challenging and exciting task. Video streams can contain a vast amount of data and extraneous content. Existing approaches tend to overlook an important aspect of prioritizing video frames that are most relevant to ongoing viewer interactions. This prioritization is crucial for producing contextually appropriate comments. To address this gap, we introduce a novel Semantic Frame Aggregation-based Transformer (SFAT) model for live video comment generation. This method not only leverages CLIP's visual-text multimodal knowledge to generate comments but also assigns weights to video frames based on their semantic relevance to ongoing viewer conversation. It employs an efficient weighted sum of frames technique to emphasize informative frames while focusing less on irrelevant ones. Finally, our comment decoder with a cross-attention mechanism that attends to each modality ensures that the generated comment reflects contextual cues from both chats and video. Furthermore, to address the limitations of existing datasets, which predominantly focus on Chinese-language content with limited video categories, we have constructed a large scale, diverse, multimodal English video comments dataset. Extracted from Twitch, this dataset covers 11 video categories, totaling 438 hours and 3.2 million comments. We demonstrate the effectiveness of our SFAT model by comparing it to existing methods for generating comments from live video and ongoing dialogue contexts. 

---
# RepV: Safety-Separable Latent Spaces for Scalable Neurosymbolic Plan Verification 

**Authors**: Yunhao Yang, Neel P. Bhatt, Pranay Samineni, Rohan Siva, Zhanyang Wang, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26935)  

**Abstract**: As AI systems migrate to safety-critical domains, verifying that their actions comply with well-defined rules remains a challenge. Formal methods provide provable guarantees but demand hand-crafted temporal-logic specifications, offering limited expressiveness and accessibility. Deep learning approaches enable evaluation of plans against natural-language constraints, yet their opaque decision process invites misclassifications with potentially severe consequences. We introduce RepV, a neurosymbolic verifier that unifies both views by learning a latent space where safe and unsafe plans are linearly separable. Starting from a modest seed set of plans labeled by an off-the-shelf model checker, RepV trains a lightweight projector that embeds each plan, together with a language model-generated rationale, into a low-dimensional space; a frozen linear boundary then verifies compliance for unseen natural-language rules in a single forward pass.
Beyond binary classification, RepV provides a probabilistic guarantee on the likelihood of correct verification based on its position in the latent space. This guarantee enables a guarantee-driven refinement of the planner, improving rule compliance without human annotations. Empirical evaluations show that RepV improves compliance prediction accuracy by up to 15% compared to baseline methods while adding fewer than 0.2M parameters. Furthermore, our refinement framework outperforms ordinary fine-tuning baselines across various planning domains. These results show that safety-separable latent spaces offer a scalable, plug-and-play primitive for reliable neurosymbolic plan verification. Code and data are available at: this https URL. 

---
# The Denario project: Deep knowledge AI agents for scientific discovery 

**Authors**: Francisco Villaescusa-Navarro, Boris Bolliet, Pablo Villanueva-Domingo, Adrian E. Bayer, Aidan Acquah, Chetana Amancharla, Almog Barzilay-Siegal, Pablo Bermejo, Camille Bilodeau, Pablo Cárdenas Ramírez, Miles Cranmer, Urbano L. França, ChangHoon Hahn, Yan-Fei Jiang, Raul Jimenez, Jun-Young Lee, Antonio Lerario, Osman Mamun, Thomas Meier, Anupam A. Ojha, Pavlos Protopapas, Shimanto Roy, David N. Spergel, Pedro Tarancón-Álvarez, Ujjwal Tiwari, Matteo Viel, Digvijay Wadekar, Chi Wang, Bonny Y. Wang, Licong Xu, Yossi Yovel, Shuwen Yue, Wen-Han Zhou, Qiyao Zhu, Jiajun Zou, Íñigo Zubeldia  

**Link**: [PDF](https://arxiv.org/pdf/2510.26887)  

**Abstract**: We present Denario, an AI multi-agent system designed to serve as a scientific research assistant. Denario can perform many different tasks, such as generating ideas, checking the literature, developing research plans, writing and executing code, making plots, and drafting and reviewing a scientific paper. The system has a modular architecture, allowing it to handle specific tasks, such as generating an idea, or carrying out end-to-end scientific analysis using Cmbagent as a deep-research backend. In this work, we describe in detail Denario and its modules, and illustrate its capabilities by presenting multiple AI-generated papers generated by it in many different scientific disciplines such as astrophysics, biology, biophysics, biomedical informatics, chemistry, material science, mathematical physics, medicine, neuroscience and planetary science. Denario also excels at combining ideas from different disciplines, and we illustrate this by showing a paper that applies methods from quantum physics and machine learning to astrophysical data. We report the evaluations performed on these papers by domain experts, who provided both numerical scores and review-like feedback. We then highlight the strengths, weaknesses, and limitations of the current system. Finally, we discuss the ethical implications of AI-driven research and reflect on how such technology relates to the philosophy of science. We publicly release the code at this https URL. A Denario demo can also be run directly on the web at this https URL, and the full app will be deployed on the cloud. 

---
# Evaluating Perspectival Biases in Cross-Modal Retrieval 

**Authors**: Teerapol Saengsukhiran, Peerawat Chomphooyod, Narabodee Rodjananant, Chompakorn Chaksangchaichot, Patawee Prakrankamanant, Witthawin Sripheanpol, Pak Lovichit, SarChaksaana Nutanong, Ekapol Chuangsuwanich  

**Link**: [PDF](https://arxiv.org/pdf/2510.26861)  

**Abstract**: Multimodal retrieval systems are expected to operate in a semantic space, agnostic to the language or cultural origin of the query. In practice, however, retrieval outcomes systematically reflect perspectival biases: deviations shaped by linguistic prevalence and cultural associations. We study two such biases. First, prevalence bias refers to the tendency to favor entries from prevalent languages over semantically faithful entries in image-to-text retrieval. Second, association bias refers to the tendency to favor images culturally associated with the query over semantically correct ones in text-to-image retrieval. Results show that explicit alignment is a more effective strategy for mitigating prevalence bias. However, association bias remains a distinct and more challenging problem. These findings suggest that achieving truly equitable multimodal systems requires targeted strategies beyond simple data scaling and that bias arising from cultural association may be treated as a more challenging problem than one arising from linguistic prevalence. 

---
# CATArena: Evaluation of LLM Agents through Iterative Tournament Competitions 

**Authors**: Lingyue Fu, Xin Ding, Yaoming Zhu, Shao Zhang, Lin Qiu, Weiwen Liu, Weinan Zhang, Xuezhi Cao, Xunliang Cai, Jiaxin Ding, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26852)  

**Abstract**: Large Language Model (LLM) agents have evolved from basic text generation to autonomously completing complex tasks through interaction with external tools. However, current benchmarks mainly assess end-to-end performance in fixed scenarios, restricting evaluation to specific skills and suffering from score saturation and growing dependence on expert annotation as agent capabilities improve. In this work, we emphasize the importance of learning ability, including both self-improvement and peer-learning, as a core driver for agent evolution toward human-level intelligence. We propose an iterative, competitive peer-learning framework, which allows agents to refine and optimize their strategies through repeated interactions and feedback, thereby systematically evaluating their learning capabilities. To address the score saturation issue in current benchmarks, we introduce CATArena, a tournament-style evaluation platform featuring four diverse board and card games with open-ended scoring. By providing tasks without explicit upper score limits, CATArena enables continuous and dynamic evaluation of rapidly advancing agent capabilities. Experimental results and analyses involving both minimal and commercial code agents demonstrate that CATArena provides reliable, stable, and scalable benchmarking for core agent abilities, particularly learning ability and strategy coding. 

---
# Broken-Token: Filtering Obfuscated Prompts by Counting Characters-Per-Token 

**Authors**: Shaked Zychlinski, Yuval Kainan  

**Link**: [PDF](https://arxiv.org/pdf/2510.26847)  

**Abstract**: Large Language Models (LLMs) are susceptible to jailbreak attacks where malicious prompts are disguised using ciphers and character-level encodings to bypass safety guardrails. While these guardrails often fail to interpret the encoded content, the underlying models can still process the harmful instructions. We introduce CPT-Filtering, a novel, model-agnostic with negligible-costs and near-perfect accuracy guardrail technique that aims to mitigate these attacks by leveraging the intrinsic behavior of Byte-Pair Encoding (BPE) tokenizers. Our method is based on the principle that tokenizers, trained on natural language, represent out-of-distribution text, such as ciphers, using a significantly higher number of shorter tokens. Our technique uses a simple yet powerful artifact of using language models: the average number of Characters Per Token (CPT) in the text. This approach is motivated by the high compute cost of modern methods - relying on added modules such as dedicated LLMs or perplexity models. We validate our approach across a large dataset of over 100,000 prompts, testing numerous encoding schemes with several popular tokenizers. Our experiments demonstrate that a simple CPT threshold robustly identifies encoded text with high accuracy, even for very short inputs. CPT-Filtering provides a practical defense layer that can be immediately deployed for real-time text filtering and offline data curation. 

---
