# IndexRAG: Bridging Facts for Cross-Document Reasoning at Index Time 

**Authors**: Zhenghua Bao, Yi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16415)  

**Abstract**: Multi-hop question answering (QA) requires reasoning across multiple documents, yet existing retrieval-augmented generation (RAG) approaches address this either through graph-based methods requiring additional online processing or iterative multi-step reasoning. We present IndexRAG, a novel approach that shifts cross-document reasoning from online inference to offline indexing. IndexRAG identifies bridge entities shared across documents and generates bridging facts as independently retrievable units, requiring no additional training or fine-tuning. Experiments on three widely-used multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue) show that IndexRAG improves F1 over Naive RAG by 4.6 points on average, while requiring only single-pass retrieval and a single LLM call at inference time. When combined with IRCoT, IndexRAG outperforms all graph-based baselines on average, including HippoRAG and FastGraphRAG, while relying solely on flat retrieval. Our code will be released upon acceptance. 

---
# Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights 

**Authors**: Yi Chen, Daiwei Chen, Sukrut Madhav Chikodikar, Caitlyn Heqi Yin, Ramya Korlakai Vinayak  

**Link**: [PDF](https://arxiv.org/pdf/2603.16817)  

**Abstract**: Large language models (LLMs) frequently hallucinate, limiting their reliability in knowledge-intensive applications. Retrieval-augmented generation (RAG) and conformal factuality have emerged as potential ways to address this limitation. While RAG aims to ground responses in retrieved evidence, it provides no statistical guarantee that the final output is correct. Conformal factuality filtering offers distribution-free statistical reliability by scoring and filtering atomic claims using a threshold calibrated on held-out data, however, the informativeness of the final output is not guaranteed. We systematically analyze the reliability and usefulness of conformal factuality for RAG-based LLMs across generation, scoring, calibration, robustness, and efficiency. We propose novel informativeness-aware metrics that better reflect task utility under conformal filtering. Across three benchmarks and multiple model families, we find that (i) conformal filtering suffers from low usefulness at high factuality levels due to vacuous outputs, (ii) conformal factuality guarantee is not robust to distribution shifts and distractors, highlighting the limitation that requires calibration data to closely match deployment conditions, and (iii) lightweight entailment-based verifiers match or outperform LLM-based model confidence scorers while requiring over $100\times$ fewer FLOPs. Overall, our results expose factuality-informativeness trade-offs and fragility of conformal filtering framework under distribution shifts and distractors, highlighting the need for new approaches for reliability with robustness and usefulness as key metrics, and provide actionable guidance for building RAG pipelines that are both reliable and computationally efficient. 

---
# Open-Source Reproduction and Explainability Analysis of Corrective Retrieval Augmented Generation 

**Authors**: Surya Vardhan Yalavarthi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16169)  

**Abstract**: Corrective Retrieval Augmented Generation (CRAG) improves the robustness of RAG systems by evaluating retrieved document quality and triggering corrective actions. However, the original implementation relies on proprietary components including the Google Search API and closed model weights, limiting reproducibility. In this work, we present a fully open-source reproduction of CRAG, replacing proprietary web search with the Wikipedia API and the original LLaMA-2 generator with Phi-3-mini-4k-instruct. We evaluate on PopQA and ARC-Challenge, demonstrating that our open-source pipeline achieves comparable performance to the original system. Furthermore, we contribute the first explainability analysis of CRAG's T5-based retrieval evaluator using SHAP, revealing that the evaluator primarily relies on named entity alignment rather than semantic similarity. Our analysis identifies key failure modes including domain transfer limitations on science questions. All code and results are available at this https URL. 

---
# ExpressMind: A Multimodal Pretrained Large Language Model for Expressway Operation 

**Authors**: Zihe Wang, Yihuan Wang, Haiyang Yu. Zhiyong Cui, Xiaojian Liao, Chengcheng Wang, Yonglin Tian, Yongxin Tong  

**Link**: [PDF](https://arxiv.org/pdf/2603.16495)  

**Abstract**: The current expressway operation relies on rule-based and isolated models, which limits the ability to jointly analyze knowledge across different systems. Meanwhile, Large Language Models (LLMs) are increasingly applied in intelligent transportation, advancing traffic models from algorithmic to cognitive intelligence. However, general LLMs are unable to effectively understand the regulations and causal relationships of events in unconventional scenarios in the expressway field. Therefore, this paper constructs a pre-trained multimodal large language model (MLLM) for expressways, ExpressMind, which serves as the cognitive core for intelligent expressway operations. This paper constructs the industry's first full-stack expressway dataset, encompassing traffic knowledge texts, emergency reasoning chains, and annotated video events to overcome data scarcity. This paper proposes a dual-layer LLM pre-training paradigm based on self-supervised training and unsupervised learning. Additionally, this study introduces a Graph-Augmented RAG framework to dynamically index the expressway knowledge base. To enhance reasoning for expressway incident response strategies, we develop a RL-aligned Chain-of-Thought (RL-CoT) mechanism that enforces consistency between model reasoning and expert problem-solving heuristics for incident handling. Finally, ExpressMind integrates a cross-modal encoder to align the dynamic feature sequences under the visual and textual channels, enabling it to understand traffic scenes in both video and image modalities. Extensive experiments on our newly released multi-modal expressway benchmark demonstrate that ExpressMind comprehensively outperforms existing baselines in event detection, safety response generation, and complex traffic analysis. The code and data are available at: this https URL. 

---
# Exploring different approaches to customize language models for domain-specific text-to-code generation 

**Authors**: Luís Freire, Fernanda A. Andaló, Nicki Skafte Detlefsen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16526)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in generating executable code from natural language descriptions. However, general-purpose models often struggle in specialized programming contexts where domain-specific libraries, APIs, or conventions must be used. Customizing smaller open-source models offers a cost-effective alternative to relying on large proprietary systems. In this work, we investigate how smaller language models can be adapted for domain-specific code generation using synthetic datasets. We construct datasets of programming exercises across three domains within the Python ecosystem: general Python programming, Scikit-learn machine learning workflows, and OpenCV-based computer vision tasks. Using these datasets, we evaluate three customization strategies: few-shot prompting, retrieval-augmented generation (RAG), and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA). Performance is evaluated using both benchmark-based metrics and similarity-based metrics that measure alignment with domain-specific code. Our results show that prompting-based approaches such as few-shot learning and RAG can improve domain relevance in a cost-effective manner, although their impact on benchmark accuracy is limited. In contrast, LoRA-based fine-tuning consistently achieves higher accuracy and stronger domain alignment across most tasks. These findings highlight practical trade-offs between flexibility, computational cost, and performance when adapting smaller language models for specialized programming tasks. 

---
# GSI Agent: Domain Knowledge Enhancement for Large Language Models in Green Stormwater Infrastructure 

**Authors**: Shaohuang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15643)  

**Abstract**: Green Stormwater Infrastructure (GSI) systems, such as permeable pavement, rain gardens, and bioretention facilities, require continuous inspection and maintenance to ensure long-term perfor- mance. However, domain knowledge about GSI is often scattered across municipal manuals, regula- tory documents, and inspection forms. As a result, non-expert users and maintenance staff may strug- gle to obtain reliable and actionable guidance from field observations. Although Large Language Models (LLMs) have demonstrated strong general reasoning and language generation capabilities, they often lack domain-specific knowledge and may produce inaccurate or hallucinated answers in engineering scenarios. This limitation restricts their direct application to professional infrastructure tasks. In this paper, we propose GSI Agent, a domain-enhanced LLM framework designed to im- prove performance in GSI-related tasks. Our approach integrates three complementary strategies: (1) supervised fine-tuning (SFT) on a curated GSI instruction dataset, (2) retrieval-augmented gen- eration (RAG) over an internal GSI knowledge base constructed from municipal documents, and (3) an agent-based reasoning pipeline that coordinates retrieval, context integration, and structured response generation. We also construct a new GSI Dataset aligned with real-world GSI inspection and maintenance scenarios. Experimental results show that our framework significantly improves domain-specific performance while maintaining general knowledge capability. On the GSI dataset, BLEU-4 improves from 0.090 to 0.307, while performance on the common knowledge dataset re- mains stable (0.304 vs. 0.305). These results demonstrate that systematic domain knowledge en- hancement can effectively adapt general-purpose LLMs to professional infrastructure applications. 

---
# Toward Experimentation-as-a-Service in 5G/6G: The Plaza6G Prototype for AI-Assisted Trials 

**Authors**: Sergio Barrachina-Muñoz, Marc Carrascosa-Zamacois, Horacio Bleda, Umair Riaz, Yasir Maqsood, Xavier Calle, Selva Vía, Miquel Payaró, Josep Mangues-Bafalluy  

**Link**: [PDF](https://arxiv.org/pdf/2603.16356)  

**Abstract**: This paper presents Plaza6G, the first operational Experiment-as-a-Service (ExaS) platform unifying cloud resources with next-generation wireless infrastructure. Developed at CTTC in Barcelona, Plaza6G integrates GPU-accelerated compute clusters, multiple 5G cores, both open-source (e.g., Free5GC) and commercial (e.g., Cumucore), programmable RANs, and physical or emulated user equipment under unified orchestration. In Plaza6G, the experiment design requires minimal expertise as it is expressed in natural language via a web portal or a REST API. The web portal and REST API are enhanced with a Large Language Model (LLM)-based assistant, which employs retrieval-augmented generation (RAG) for up-to-date experiment knowledge and Low-Rank Adaptation (LoRA) for continuous domain fine-tuning. Over-the-air (OTA) trials leverage a four-chamber anechoic facility and a dual-site outdoor 5G network operating in sub-6~GHz and mmWave bands. Demonstrations include automated CI/CD integration with sub-ten-minute setup and interactive OTA testing under programmable propagation conditions. Machine-readable experiment descriptors ensure reproducibility, while future work targets policy-aware orchestration, safety validation, and federated testbed integration toward open, reproducible wireless experimentation. 

---
