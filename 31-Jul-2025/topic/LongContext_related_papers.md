# NeedleChain: Measuring Intact Long-Context Reasoning Capability of Large Language Models 

**Authors**: Hyeonseok Moon, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.22411)  

**Abstract**: The Needle-in-a-Haystack (NIAH) benchmark is widely used to evaluate Large Language Models' (LLMs) ability to understand long contexts (LC). It evaluates the capability to identify query-relevant context within extensive query-irrelevant passages. Although this method serves as a widely accepted standard for evaluating long-context understanding, our findings suggest it may overestimate the true LC capability of LLMs. We demonstrate that even state-of-the-art models such as GPT-4o struggle to intactly incorporate given contexts made up of solely query-relevant ten sentences. In response, we introduce a novel benchmark, \textbf{NeedleChain}, where the context consists entirely of query-relevant information, requiring the LLM to fully grasp the input to answer correctly. Our benchmark allows for flexible context length and reasoning order, offering a more comprehensive analysis of LLM performance. Additionally, we propose an extremely simple yet compelling strategy to improve LC understanding capability of LLM: ROPE Contraction. Our experiments with various advanced LLMs reveal a notable disparity between their ability to process large contexts and their capacity to fully understand them. Source code and datasets are available at this https URL 

---
# CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records 

**Authors**: Dongchen Li, Jitao Liang, Wei Li, Xiaoyu Wang, Longbing Cao, Kun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22533)  

**Abstract**: Large Language Models (LLMs) hold significant promise for improving clinical decision support and reducing physician burnout by synthesizing complex, longitudinal cancer Electronic Health Records (EHRs). However, their implementation in this critical field faces three primary challenges: the inability to effectively process the extensive length and multilingual nature of patient records for accurate temporal analysis; a heightened risk of clinical hallucination, as conventional grounding techniques such as Retrieval-Augmented Generation (RAG) do not adequately incorporate process-oriented clinical guidelines; and unreliable evaluation metrics that hinder the validation of AI systems in oncology. To address these issues, we propose CliCARE, a framework for Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records. The framework operates by transforming unstructured, longitudinal EHRs into patient-specific Temporal Knowledge Graphs (TKGs) to capture long-range dependencies, and then grounding the decision support process by aligning these real-world patient trajectories with a normative guideline knowledge graph. This approach provides oncologists with evidence-grounded decision support by generating a high-fidelity clinical summary and an actionable recommendation. We validated our framework using large-scale, longitudinal data from a private Chinese cancer dataset and the public English MIMIC-IV dataset. In these diverse settings, CliCARE significantly outperforms strong baselines, including leading long-context LLMs and Knowledge Graph-enhanced RAG methods. The clinical validity of our results is supported by a robust evaluation protocol, which demonstrates a high correlation with assessments made by expert oncologists. 

---
