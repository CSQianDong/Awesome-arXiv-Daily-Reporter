# Ethical AI prompt recommendations in large language models using collaborative filtering 

**Authors**: Jordan Nelson, Almas Baimagambetov, Konstantinos Avgerinakis, Nikolaos Polatidis  

**Link**: [PDF](https://arxiv.org/pdf/2510.06924)  

**Abstract**: As large language models (LLMs) shape AI development, ensuring ethical prompt recommendations is crucial. LLMs offer innovation but risk bias, fairness issues, and accountability concerns. Traditional oversight methods struggle with scalability, necessitating dynamic solutions. This paper proposes using collaborative filtering, a technique from recommendation systems, to enhance ethical prompt selection. By leveraging user interactions, it promotes ethical guidelines while reducing bias. Contributions include a synthetic dataset for prompt recommendations and the application of collaborative filtering. The work also tackles challenges in ethical AI, such as bias mitigation, transparency, and preventing unethical prompt engineering. 

---
# M3Retrieve: Benchmarking Multimodal Retrieval for Medicine 

**Authors**: Arkadeep Acharya, Akash Ghosh, Pradeepika Verma, Kitsuchart Pasupa, Sriparna Saha, Priti Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.06888)  

**Abstract**: With the increasing use of RetrievalAugmented Generation (RAG), strong retrieval models have become more important than ever. In healthcare, multimodal retrieval models that combine information from both text and images offer major advantages for many downstream tasks such as question answering, cross-modal retrieval, and multimodal summarization, since medical data often includes both formats. However, there is currently no standard benchmark to evaluate how well these models perform in medical settings. To address this gap, we introduce M3Retrieve, a Multimodal Medical Retrieval Benchmark. M3Retrieve, spans 5 domains,16 medical fields, and 4 distinct tasks, with over 1.2 Million text documents and 164K multimodal queries, all collected under approved licenses. We evaluate leading multimodal retrieval models on this benchmark to explore the challenges specific to different medical specialities and to understand their impact on retrieval performance. By releasing M3Retrieve, we aim to enable systematic evaluation, foster model innovation, and accelerate research toward building more capable and reliable multimodal retrieval systems for medical applications. The dataset and the baselines code are available in this github page this https URL. 

---
# Crossing Domains without Labels: Distant Supervision for Term Extraction 

**Authors**: Elena Senger, Yuri Campbell, Rob van der Goot, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2510.06838)  

**Abstract**: Automatic Term Extraction (ATE) is a critical component in downstream NLP tasks such as document tagging, ontology construction and patent analysis. Current state-of-the-art methods require expensive human annotation and struggle with domain transfer, limiting their practical deployment. This highlights the need for more robust, scalable solutions and realistic evaluation settings. To address this, we introduce a comprehensive benchmark spanning seven diverse domains, enabling performance evaluation at both the document- and corpus-levels. Furthermore, we propose a robust LLM-based model that outperforms both supervised cross-domain encoder models and few-shot learning baselines and performs competitively with its GPT-4o teacher on this benchmark. The first step of our approach is generating psuedo-labels with this black-box LLM on general and scientific domains to ensure generalizability. Building on this data, we fine-tune the first LLMs for ATE. To further enhance document-level consistency, oftentimes needed for downstream tasks, we introduce lightweight post-hoc heuristics. Our approach exceeds previous approaches on 5/7 domains with an average improvement of 10 percentage points. We release our dataset and fine-tuned models to support future research in this area. 

---
# Reproducing and Extending Causal Insights Into Term Frequency Computation in Neural Rankers 

**Authors**: Cile van Marken, Roxana Petcu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06728)  

**Abstract**: Neural ranking models have shown outstanding performance across a variety of tasks, such as document retrieval, re-ranking, question answering and conversational retrieval. However, the inner decision process of these models remains largely unclear, especially as models increase in size. Most interpretability approaches, such as probing, focus on correlational insights rather than establishing causal relationships. The paper 'Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models' by Chen et al. addresses this gap by introducing a framework for activation patching - a causal interpretability method - in the information retrieval domain, offering insights into how neural retrieval models compute document relevance. The study demonstrates that neural ranking models not only capture term-frequency information, but also that these representations can be localized to specific components of the model, such as individual attention heads or layers. This paper aims to reproduce the findings by Chen et al. and to further explore the presence of pre-defined retrieval axioms in neural IR models. We validate the main claims made by Chen et al., and extend the framework to include an additional term-frequency axiom, which states that the impact of increasing query term frequency on document ranking diminishes as the frequency becomes higher. We successfully identify a group of attention heads that encode this axiom and analyze their behavior to give insight into the inner decision-making process of neural ranking models. 

---
# Can We Hide Machines in the Crowd? Quantifying Equivalence in LLM-in-the-loop Annotation Tasks 

**Authors**: Jiaman He, Zikang Leng, Dana McKay, Damiano Spina, Johanne R. Trippas  

**Link**: [PDF](https://arxiv.org/pdf/2510.06658)  

**Abstract**: Many evaluations of large language models (LLMs) in text annotation focus primarily on the correctness of the output, typically comparing model-generated labels to human-annotated ``ground truth'' using standard performance metrics. In contrast, our study moves beyond effectiveness alone. We aim to explore how labeling decisions -- by both humans and LLMs -- can be statistically evaluated across individuals. Rather than treating LLMs purely as annotation systems, we approach LLMs as an alternative annotation mechanism that may be capable of mimicking the subjective judgments made by humans. To assess this, we develop a statistical evaluation method based on Krippendorff's $\alpha$, paired bootstrapping, and the Two One-Sided t-Tests (TOST) equivalence test procedure. This evaluation method tests whether an LLM can blend into a group of human annotators without being distinguishable.
We apply this approach to two datasets -- MovieLens 100K and PolitiFact -- and find that the LLM is statistically indistinguishable from a human annotator in the former ($p = 0.004$), but not in the latter ($p = 0.155$), highlighting task-dependent differences. It also enables early evaluation on a small sample of human data to inform whether LLMs are suitable for large-scale annotation in a given application. 

---
# LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations 

**Authors**: Boyuan Long, Yueqi Wang, Hiloni Mehta, Mick Zomnir, Omkar Pathak, Changping Meng, Ruolin Jia, Yajun Peng, Dapeng Hong, Xia Wu, Mingyan Gao, Onkar Dalal, Ningren Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.06657)  

**Abstract**: This paper presents a case study on deploying Large Language Models (LLMs) as an advanced "annotation" mechanism to achieve nuanced content understanding (e.g., discerning content "vibe") at scale within a large-scale industrial short-form video recommendation system. Traditional machine learning classifiers for content understanding face protracted development cycles and a lack of deep, nuanced comprehension. The "LLM-as-annotators" approach addresses these by significantly shortening development times and enabling the annotation of subtle attributes. This work details an end-to-end workflow encompassing: (1) iterative definition and robust evaluation of target attributes, refined by offline metrics and online A/B testing; (2) scalable offline bulk annotation of video corpora using LLMs with multimodal features, optimized inference, and knowledge distillation for broad application; and (3) integration of these rich annotations into the online recommendation serving system, for example, through personalized restrict retrieval. Experimental results demonstrate the efficacy of this approach, with LLMs outperforming human raters in offline annotation quality for nuanced attributes and yielding significant improvements of user participation and satisfied consumption in online A/B tests. The study provides insights into designing and scaling production-level LLM pipelines for rich content evaluation, highlighting the adaptability and benefits of LLM-generated nuanced understanding for enhancing content discovery, user satisfaction, and the overall effectiveness of modern recommendation systems. 

---
# Towards Reliable Retrieval in RAG Systems for Large Legal Datasets 

**Authors**: Markus Reuter, Tobias Lingenberg, Rūta Liepiņa, Francesca Lagioia, Marco Lippi, Giovanni Sartor, Andrea Passerini, Burcu Sayin  

**Link**: [PDF](https://arxiv.org/pdf/2510.06999)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a promising approach to mitigate hallucinations in Large Language Models (LLMs) for legal applications, but its reliability is critically dependent on the accuracy of the retrieval step. This is particularly challenging in the legal domain, where large databases of structurally similar documents often cause retrieval systems to fail. In this paper, we address this challenge by first identifying and quantifying a critical failure mode we term Document-Level Retrieval Mismatch (DRM), where the retriever selects information from entirely incorrect source documents. To mitigate DRM, we investigate a simple and computationally efficient technique which we refer to as Summary-Augmented Chunking (SAC). This method enhances each text chunk with a document-level synthetic summary, thereby injecting crucial global context that would otherwise be lost during a standard chunking process. Our experiments on a diverse set of legal information retrieval tasks show that SAC greatly reduces DRM and, consequently, also improves text-level retrieval precision and recall. Interestingly, we find that a generic summarization strategy outperforms an approach that incorporates legal expert domain knowledge to target specific legal elements. Our work provides evidence that this practical, scalable, and easily integrable technique enhances the reliability of RAG systems when applied to large-scale legal document datasets. 

---
# Spiral Model Technique For Data Science & Machine Learning Lifecycle 

**Authors**: Rohith Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2510.06987)  

**Abstract**: Analytics play an important role in modern business. Companies adapt data science lifecycles to their culture to seek productivity and improve their competitiveness among others. Data science lifecycles are fairly an important contributing factor to start and end a project that are data dependent. Data science and Machine learning life cycles comprises of series of steps that are involved in a project. A typical life cycle states that it is a linear or cyclical model that revolves around. It is mostly depicted that it is possible in a traditional data science life cycle to start the process again after reaching the end of cycle. This paper suggests a new technique to incorporate data science life cycle to business problems that have a clear end goal. A new technique called spiral technique is introduced to emphasize versatility, agility and iterative approach to business processes. 

---
# Exposing Citation Vulnerabilities in Generative Engines 

**Authors**: Riku Mochizuki, Shusuke Komatsu, Souta Noguchi, Kazuto Ataka  

**Link**: [PDF](https://arxiv.org/pdf/2510.06823)  

**Abstract**: We analyze answers generated by generative engines (GEs) from the perspectives of citation publishers and the content-injection barrier, defined as the difficulty for attackers to manipulate answers to user prompts by placing malicious content on the web. GEs integrate two functions: web search and answer generation that cites web pages using large language models. Because anyone can publish information on the web, GEs are vulnerable to poisoning attacks. Existing studies of citation evaluation focus on how faithfully answer content reflects cited sources, leaving unexamined which web sources should be selected as citations to defend against poisoning attacks. To fill this gap, we introduce evaluation criteria that assess poisoning threats using the citation information contained in answers. Our criteria classify the publisher attributes of citations to estimate the content-injection barrier thereby revealing the threat of poisoning attacks in current GEs. We conduct experiments in political domains in Japan and the United States (U.S.) using our criteria and show that citations from official party websites (primary sources) are approximately \(25\%\)--\(45\%\) in the U.S. and \(60\%\)--\(65\%\) in Japan, indicating that U.S. political answers are at higher risk of poisoning attacks. We also find that sources with low content-injection barriers are frequently cited yet are poorly reflected in answer content. To mitigate this threat, we discuss how publishers of primary sources can increase exposure of their web content in answers and show that well-known techniques are limited by language differences. 

---
# Overview of the Plagiarism Detection Task at PAN 2025 

**Authors**: André Greiner-Petter, Maik Fröbe, Jan Philip Wahle, Terry Ruas, Bela Gipp, Akiko Aizawa, Martin Potthast  

**Link**: [PDF](https://arxiv.org/pdf/2510.06805)  

**Abstract**: The generative plagiarism detection task at PAN 2025 aims at identifying automatically generated textual plagiarism in scientific articles and aligning them with their respective sources. We created a novel large-scale dataset of automatically generated plagiarism using three large language models: Llama, DeepSeek-R1, and Mistral. In this task overview paper, we outline the creation of this dataset, summarize and compare the results of all participants and four baselines, and evaluate the results on the last plagiarism detection task from PAN 2015 in order to interpret the robustness of the proposed approaches. We found that the current iteration does not invite a large variety of approaches as naive semantic similarity approaches based on embedding vectors provide promising results of up to 0.8 recall and 0.5 precision. In contrast, most of these approaches underperform significantly on the 2015 dataset, indicating a lack in generalizability. 

---
# Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization 

**Authors**: Tiancheng Xing, Jerry Li, Yixuan Du, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06732)  

**Abstract**: Large language models (LLMs) are increasingly used as rerankers in information retrieval, yet their ranking behavior can be steered by small, natural-sounding prompts. To expose this vulnerability, we present Rank Anything First (RAF), a two-stage token optimization method that crafts concise textual perturbations to consistently promote a target item in LLM-generated rankings while remaining hard to detect. Stage 1 uses Greedy Coordinate Gradient to shortlist candidate tokens at the current position by combining the gradient of the rank-target with a readability score; Stage 2 evaluates those candidates under exact ranking and readability losses using an entropy-based dynamic weighting scheme, and selects a token via temperature-controlled sampling. RAF generates ranking-promoting prompts token-by-token, guided by dual objectives: maximizing ranking effectiveness and preserving linguistic naturalness. Experiments across multiple LLMs show that RAF significantly boosts the rank of target items using naturalistic language, with greater robustness than existing methods in both promoting target items and maintaining naturalness. These findings underscore a critical security implication: LLM-based reranking is inherently susceptible to adversarial manipulation, raising new challenges for the trustworthiness and robustness of modern retrieval systems. Our code is available at: this https URL. 

---
