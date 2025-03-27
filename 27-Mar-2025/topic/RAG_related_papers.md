# MCTS-RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo Tree Search 

**Authors**: Yunhai Hu, Yilun Zhao, Chen Zhao, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2503.20757)  

**Abstract**: We introduce MCTS-RAG, a novel approach that enhances the reasoning capabilities of small language models on knowledge-intensive tasks by leveraging retrieval-augmented generation (RAG) to provide relevant context and Monte Carlo Tree Search (MCTS) to refine reasoning paths. MCTS-RAG dynamically integrates retrieval and reasoning through an iterative decision-making process. Unlike standard RAG methods, which typically retrieve information independently from reasoning and thus integrate knowledge suboptimally, or conventional MCTS reasoning, which depends solely on internal model knowledge without external facts, MCTS-RAG combines structured reasoning with adaptive retrieval. This integrated approach enhances decision-making, reduces hallucinations, and ensures improved factual accuracy and response consistency. The experimental results on multiple reasoning and knowledge-intensive datasets datasets (i.e., ComplexWebQA, GPQA, and FoolMeTwice) show that our method enables small-scale LMs to achieve performance comparable to frontier LLMs like GPT-4o by effectively scaling inference-time compute, setting a new standard for reasoning in small-scale models. 

---
# Dewey Long Context Embedding Model: A Technical Report 

**Authors**: Dun Zhang, Panxiang Zou, Yudong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.20376)  

**Abstract**: This technical report presents the training methodology and evaluation results of the open-source dewey_en_beta embedding model. The increasing demand for retrieval-augmented generation (RAG) systems and the expanding context window capabilities of large language models (LLMs) have created critical challenges for conventional embedding models. Current approaches often struggle to maintain semantic coherence when processing documents exceeding typical sequence length limitations, significantly impacting retrieval performance in knowledge-intensive applications. This paper presents dewey_en_beta, a novel text embedding model that achieves excellent performance on MTEB (Eng, v2) and LongEmbed benchmark while supporting 128K token sequences. Our technical contribution centers on chunk alignment training, an innovative methodology that enables the simultaneous generation of localized chunk embeddings and global document-level representations through distillation. Information regarding the model release can be found at this https URL. 

---
# RALLRec+: Retrieval Augmented Large Language Model Recommendation with Reasoning 

**Authors**: Sichun Luo, Jian Xu, Xiaojie Zhang, Linrong Wang, Sicong Liu, Hanxu Hou, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.20430)  

**Abstract**: Large Language Models (LLMs) have been integrated into recommender systems to enhance user behavior comprehension. The Retrieval Augmented Generation (RAG) technique is further incorporated into these systems to retrieve more relevant items and improve system performance. However, existing RAG methods have two shortcomings. \textit{(i)} In the \textit{retrieval} stage, they rely primarily on textual semantics and often fail to incorporate the most relevant items, thus constraining system effectiveness. \textit{(ii)} In the \textit{generation} stage, they lack explicit chain-of-thought reasoning, further limiting their potential.
In this paper, we propose Representation learning and \textbf{R}easoning empowered retrieval-\textbf{A}ugmented \textbf{L}arge \textbf{L}anguage model \textbf{Rec}ommendation (RALLRec+). Specifically, for the retrieval stage, we prompt LLMs to generate detailed item descriptions and perform joint representation learning, combining textual and collaborative signals extracted from the LLM and recommendation models, respectively. To account for the time-varying nature of user interests, we propose a simple yet effective reranking method to capture preference dynamics. For the generation phase, we first evaluate reasoning LLMs on recommendation tasks, uncovering valuable insights. Then we introduce knowledge-injected prompting and consistency-based merging approach to integrate reasoning LLMs with general-purpose LLMs, enhancing overall performance. Extensive experiments on three real world datasets validate our method's effectiveness. 

---
