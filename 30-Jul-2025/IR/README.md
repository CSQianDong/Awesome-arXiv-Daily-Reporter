# The Curious Case of High-Dimensional Indexing as a File Structure: A Case Study of eCP-FS 

**Authors**: Omar Shahbaz Khan, Gylfi Þór Guðmundsson, Björn Þór Jónsson  

**Link**: [PDF](https://arxiv.org/pdf/2507.21939)  

**Abstract**: Modern analytical pipelines routinely deploy multiple deep learning and retrieval models that rely on approximate nearest-neighbor (ANN) indexes to support efficient similarity-based search. While many state-of-the-art ANN-indexes are memory-based (e.g., HNSW and IVF), using multiple ANN indexes creates a competition for limited GPU/CPU memory resources, which in turn necessitates disk-based index structures (e.g., DiskANN or eCP). In typical index implementations, the main component is a complex data structure that is serialized to disk and is read either fully at startup time, for memory-based indexes, or incrementally at query time, for disk-based indexes. To visualize the index structure, or analyze its quality, complex coding is needed that is either embedded in the index implementation or replicates the code that reads the data structure. In this paper, we consider an alternative approach that maps the data structure to a file structure, using a file library, making the index easily readable for any programming language and even human-readable. The disadvantage is that the serialized index is verbose, leading to overhead of searching through the index. The question addressed in this paper is how severe this performance penalty is. To that end, this paper presents eCP-FS, a file-based implementation of eCP, a well-known disk-based ANN index. A comparison with state-of-the-art indexes shows that while eCP-FS is slower, the implementation is nevertheless somewhat competitive even when memory is not constrained. In a memory-constrained scenario, eCP-FS offers a minimal memory footprint, making it ideal for resource-constrained or multi-index environments. 

---
# Exploration on Demand: From Algorithmic Control to User Empowerment 

**Authors**: Edoardo Bianchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21884)  

**Abstract**: Recommender systems often struggle with over-specialization, which severely limits users' exposure to diverse content and creates filter bubbles that reduce serendipitous discovery. To address this fundamental limitation, this paper introduces an adaptive clustering framework with user-controlled exploration that effectively balances personalization and diversity in movie recommendations. Our approach leverages sentence-transformer embeddings to group items into semantically coherent clusters through an online algorithm with dynamic thresholding, thereby creating a structured representation of the content space. Building upon this clustering foundation, we propose a novel exploration mechanism that empowers users to control recommendation diversity by strategically sampling from less-engaged clusters, thus expanding their content horizons while preserving relevance. Experiments on the MovieLens dataset demonstrate the system's effectiveness, showing that exploration significantly reduces intra-list similarity from 0.34 to 0.26 while simultaneously increasing unexpectedness to 0.73. Furthermore, our Large Language Model-based A/B testing methodology, conducted with 300 simulated users, reveals that 72.7% of long-term users prefer exploratory recommendations over purely exploitative ones, providing strong evidence for the system's ability to promote meaningful content discovery without sacrificing user satisfaction. 

---
# Proposing a Semantic Movie Recommendation System Enhanced by ChatGPT's NLP Results 

**Authors**: Ali Fallahi, Azam Bastanfard, Amineh Amini, Hadi Saboohi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21770)  

**Abstract**: The importance of recommender systems on the web has grown, especially in the movie industry, with a vast selection of options to watch. To assist users in traversing available items and finding relevant results, recommender systems analyze operational data and investigate users' tastes and habits. Providing highly individualized suggestions can boost user engagement and satisfaction, which is one of the fundamental goals of the movie industry, significantly in online platforms. According to recent studies and research, using knowledge-based techniques and considering the semantic ideas of the textual data is a suitable way to get more appropriate results. This study provides a new method for building a knowledge graph based on semantic information. It uses the ChatGPT, as a large language model, to assess the brief descriptions of movies and extract their tone of voice. Results indicated that using the proposed method may significantly enhance accuracy rather than employing the explicit genres supplied by the publishers. 

---
# Enhancing Graph-based Recommendations with Majority-Voting LLM-Rerank Augmentation 

**Authors**: Minh-Anh Nguyen, Bao Nguyen, Ha Lan N.T., Tuan Anh Hoang, Duc-Trong Le, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2507.21563)  

**Abstract**: Recommendation systems often suffer from data sparsity caused by limited user-item interactions, which degrade their performance and amplify popularity bias in real-world scenarios. This paper proposes a novel data augmentation framework that leverages Large Language Models (LLMs) and item textual descriptions to enrich interaction data. By few-shot prompting LLMs multiple times to rerank items and aggregating the results via majority voting, we generate high-confidence synthetic user-item interactions, supported by theoretical guarantees based on the concentration of measure. To effectively leverage the augmented data in the context of a graph recommendation system, we integrate it into a graph contrastive learning framework to mitigate distributional shift and alleviate popularity bias. Extensive experiments show that our method improves accuracy and reduces popularity bias, outperforming strong baselines. 

---
# Solution for Meta KDD Cup'25: A Comprehensive Three-Step Framework for Vision Question Answering 

**Authors**: Zijian Zhang, Xiaocheng Zhang, Yang Zhou, Zhimin Lin, Peng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21520)  

**Abstract**: Vision Large Language Models (VLLMs) have improved multi-modal understanding and visual question answering (VQA), but still suffer from hallucinated answers. Multi-modal Retrieval-Augmented Generation (RAG) helps address these issues by incorporating external information, yet challenges remain in visual context comprehension, multi-source retrieval, and multi-turn interactions. To address these challenges, Meta constructed the CRAG-MM benchmark and launched the CRAG-MM Challenge at KDD Cup 2025, which consists of three tasks. This paper describes the solutions of all tasks in Meta KDD Cup'25 from BlackPearl team. We use a single model for each task, with key methods including data augmentation, RAG, reranking, and multi-task fine-tuning. Our solution achieve automatic evaluation rankings of 3rd, 3rd, and 1st on the three tasks, and win second place in Task3 after human evaluation. 

---
# Efficient Data Retrieval and Comparative Bias Analysis of Recommendation Algorithms for YouTube Shorts and Long-Form Videos 

**Authors**: Selimhan Dagtas, Mert Can Cakmak, Nitin Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2507.21467)  

**Abstract**: The growing popularity of short-form video content, such as YouTube Shorts, has transformed user engagement on digital platforms, raising critical questions about the role of recommendation algorithms in shaping user experiences. These algorithms significantly influence content consumption, yet concerns about biases, echo chambers, and content diversity persist. This study develops an efficient data collection framework to analyze YouTube's recommendation algorithms for both short-form and long-form videos, employing parallel computing and advanced scraping techniques to overcome limitations of YouTube's API. The analysis uncovers distinct behavioral patterns in recommendation algorithms across the two formats, with short-form videos showing a more immediate shift toward engaging yet less diverse content compared to long-form videos. Furthermore, a novel investigation into biases in politically sensitive topics, such as the South China Sea dispute, highlights the role of these algorithms in shaping narratives and amplifying specific viewpoints. By providing actionable insights for designing equitable and transparent recommendation systems, this research underscores the importance of responsible AI practices in the evolving digital media landscape. 

---
# RATE: An LLM-Powered Retrieval Augmented Generation Technology-Extraction Pipeline 

**Authors**: Karan Mirhosseini, Arya Aftab, Alireza Sheikh  

**Link**: [PDF](https://arxiv.org/pdf/2507.21125)  

**Abstract**: In an era of radical technology transformations, technology maps play a crucial role in enhancing decision making. These maps heavily rely on automated methods of technology extraction. This paper introduces Retrieval Augmented Technology Extraction (RATE), a Large Language Model (LLM) based pipeline for automated technology extraction from scientific literature. RATE combines Retrieval Augmented Generation (RAG) with multi-definition LLM-based validation. This hybrid method results in high recall in candidate generation alongside with high precision in candidate filtering. While the pipeline is designed to be general and widely applicable, we demonstrate its use on 678 research articles focused on Brain-Computer Interfaces (BCIs) and Extended Reality (XR) as a case study. Consequently, The validated technology terms by RATE were mapped into a co-occurrence network, revealing thematic clusters and structural features of the research landscape. For the purpose of evaluation, a gold standard dataset of technologies in 70 selected random articles had been curated by the experts. In addition, a technology extraction model based on Bidirectional Encoder Representations of Transformers (BERT) was used as a comparative method. RATE achieved F1-score of 91.27%, Significantly outperforming BERT with F1-score of 53.73%. Our findings highlight the promise of definition-driven LLM methods for technology extraction and mapping. They also offer new insights into emerging trends within the BCI-XR field. The source code is available this https URL 

---
# Affect-aware Cross-Domain Recommendation for Art Therapy via Music Preference Elicitation 

**Authors**: Bereket A. Yilma, Luis A. Leiva  

**Link**: [PDF](https://arxiv.org/pdf/2507.21120)  

**Abstract**: Art Therapy (AT) is an established practice that facilitates emotional processing and recovery through creative expression. Recently, Visual Art Recommender Systems (VA RecSys) have emerged to support AT, demonstrating their potential by personalizing therapeutic artwork recommendations. Nonetheless, current VA RecSys rely on visual stimuli for user modeling, limiting their ability to capture the full spectrum of emotional responses during preference elicitation. Previous studies have shown that music stimuli elicit unique affective reflections, presenting an opportunity for cross-domain recommendation (CDR) to enhance personalization in AT. Since CDR has not yet been explored in this context, we propose a family of CDR methods for AT based on music-driven preference elicitation. A large-scale study with 200 users demonstrates the efficacy of music-driven preference elicitation, outperforming the classic visual-only elicitation approach. Our source code, data, and models are available at this https URL 

---
# A Comprehensive Review on Harnessing Large Language Models to Overcome Recommender System Challenges 

**Authors**: Rahul Raja, Anshaj Vats, Arpita Vats, Anirban Majumder  

**Link**: [PDF](https://arxiv.org/pdf/2507.21117)  

**Abstract**: Recommender systems have traditionally followed modular architectures comprising candidate generation, multi-stage ranking, and re-ranking, each trained separately with supervised objectives and hand-engineered features. While effective in many domains, such systems face persistent challenges including sparse and noisy interaction data, cold-start problems, limited personalization depth, and inadequate semantic understanding of user and item content. The recent emergence of Large Language Models (LLMs) offers a new paradigm for addressing these limitations through unified, language-native mechanisms that can generalize across tasks, domains, and modalities. In this paper, we present a comprehensive technical survey of how LLMs can be leveraged to tackle key challenges in modern recommender systems. We examine the use of LLMs for prompt-driven candidate retrieval, language-native ranking, retrieval-augmented generation (RAG), and conversational recommendation, illustrating how these approaches enhance personalization, semantic alignment, and interpretability without requiring extensive task-specific supervision. LLMs further enable zero- and few-shot reasoning, allowing systems to operate effectively in cold-start and long-tail scenarios by leveraging external knowledge and contextual cues. We categorize these emerging LLM-driven architectures and analyze their effectiveness in mitigating core bottlenecks of conventional pipelines. In doing so, we provide a structured framework for understanding the design space of LLM-enhanced recommenders, and outline the trade-offs between accuracy, scalability, and real-time performance. Our goal is to demonstrate that LLMs are not merely auxiliary components but foundational enablers for building more adaptive, semantically rich, and user-centric recommender systems 

---
# FedFlex: Federated Learning for Diverse Netflix Recommendations 

**Authors**: Sven Lankester, Manel Slokom, Gustavo de Carvalho Bertoli, Matias Vizcaino, Emmanuelle Beauxis Aussalet, Laura Hollink  

**Link**: [PDF](https://arxiv.org/pdf/2507.21115)  

**Abstract**: Federated learning is a decentralized approach that enables collaborative model training across multiple devices while preserving data privacy. It has shown significant potential in various domains, including healthcare and personalized recommendation systems. However, most existing work on federated recommendation systems has focused primarily on improving accuracy, with limited attention to fairness and diversity. In this paper, we introduce FedFlex, a federated recommender system for Netflix-style TV series recommendations. FedFlex integrates two state-of-the-art matrix factorization algorithms for personalized fine-tuning. FedFlex also applies Maximal Marginal Relevance (MMR) to re-rank items and enhance diversity. We conduct extensive experiments comparing recommendations generated by SVD and BPR algorithms. In a live two-week user study, participants received two recommendation lists: List A, based on SVD or BPR, and List B, a re-ranked version emphasizing diversity. Participants were asked to click on the movies they were interested in watching. Our findings demonstrate that FedFlex effectively introduces diverse content, such as new genres, into recommendations without necessarily compromising user satisfaction. 

---
# Page image classification for content-specific data processing 

**Authors**: Kateryna Lutsai, Pavel Straňák  

**Link**: [PDF](https://arxiv.org/pdf/2507.21114)  

**Abstract**: Digitization projects in humanities often generate vast quantities of page images from historical documents, presenting significant challenges for manual sorting and analysis. These archives contain diverse content, including various text types (handwritten, typed, printed), graphical elements (drawings, maps, photos), and layouts (plain text, tables, forms). Efficiently processing this heterogeneous data requires automated methods to categorize pages based on their content, enabling tailored downstream analysis pipelines. This project addresses this need by developing and evaluating an image classification system specifically designed for historical document pages, leveraging advancements in artificial intelligence and machine learning. The set of categories was chosen to facilitate content-specific processing workflows, separating pages requiring different analysis techniques (e.g., OCR for text, image analysis for graphics) 

---
# AgentMaster: A Multi-Agent Conversational Framework Using A2A and MCP Protocols for Multimodal Information Retrieval and Analysis 

**Authors**: Callie C. Liao, Duoduo Liao, Sai Surya Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2507.21105)  

**Abstract**: The rise of Multi-Agent Systems (MAS) in Artificial Intelligence (AI), especially integrated with Large Language Models (LLMs), has greatly facilitated the resolution of complex tasks. However, current systems are still facing challenges of inter-agent communication, coordination, and interaction with heterogeneous tools and resources. Most recently, the Model Context Protocol (MCP) by Anthropic and Agent-to-Agent (A2A) communication protocol by Google have been introduced, and to the best of our knowledge, very few applications exist where both protocols are employed within a single MAS framework. We present a pilot study of AgentMaster, a novel modular multi-protocol MAS framework with self-implemented A2A and MCP, enabling dynamic coordination and flexible communication. Through a unified conversational interface, the system supports natural language interaction without prior technical expertise and responds to multimodal queries for tasks including information retrieval, question answering, and image analysis. Evaluation through the BERTScore F1 and LLM-as-a-Judge metric G-Eval averaged 96.3\% and 87.1\%, revealing robust inter-agent coordination, query decomposition, dynamic routing, and domain-specific, relevant responses. Overall, our proposed framework contributes to the potential capabilities of domain-specific, cooperative, and scalable conversational AI powered by MAS. 

---
# Analise Semantica Automatizada com LLM e RAG para Bulas Farmaceuticas 

**Authors**: Daniel Meireles do Rego  

**Link**: [PDF](https://arxiv.org/pdf/2507.21103)  

**Abstract**: The production of digital documents has been growing rapidly in academic, business, and health environments, presenting new challenges in the efficient extraction and analysis of unstructured information. This work investigates the use of RAG (Retrieval-Augmented Generation) architectures combined with Large-Scale Language Models (LLMs) to automate the analysis of documents in PDF format. The proposal integrates vector search techniques by embeddings, semantic data extraction and generation of contextualized natural language responses. To validate the approach, we conducted experiments with drug package inserts extracted from official public sources. The semantic queries applied were evaluated by metrics such as accuracy, completeness, response speed and consistency. The results indicate that the combination of RAG with LLMs offers significant gains in intelligent information retrieval and interpretation of unstructured technical texts. 

---
# Not Here, Go There: Analyzing Redirection Patterns on the Web 

**Authors**: Kritika Garg, Sawood Alam, Dietrich Ayala, Michele C. Weigle, Michael L. Nelson  

**Link**: [PDF](https://arxiv.org/pdf/2507.22019)  

**Abstract**: URI redirections are integral to web management, supporting structural changes, SEO optimization, and security. However, their complexities affect usability, SEO performance, and digital preservation. This study analyzed 11 million unique redirecting URIs, following redirections up to 10 hops per URI, to uncover patterns and implications of redirection practices. Our findings revealed that 50% of the URIs terminated successfully, while 50% resulted in errors, including 0.06% exceeding 10 hops. Canonical redirects, such as HTTP to HTTPS transitions, were prevalent, reflecting adherence to SEO best practices. Non-canonical redirects, often involving domain or path changes, highlighted significant web migrations, rebranding, and security risks. Notable patterns included "sink" URIs, where multiple redirects converged, ranging from traffic consolidation by global websites to deliberate "Rickrolling." The study also identified 62,000 custom 404 URIs, almost half being soft 404s, which could compromise SEO and user experience. These findings underscore the critical role of URI redirects in shaping the web while exposing challenges such as outdated URIs, server instability, and improper error handling. This research offers a detailed analysis of URI redirection practices, providing insights into their prevalence, types, and outcomes. By examining a large dataset, we highlight inefficiencies in redirection chains and examine patterns such as the use of "sink" URIs and custom error pages. This information can help webmasters, researchers, and digital archivists improve web usability, optimize resource allocation, and safeguard valuable online content. 

---
# Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on Transformer-based Embedding Vectors 

**Authors**: Patrick Iff, Paul Bruegger, Marcin Chrapek, Maciej Besta, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2507.21989)  

**Abstract**: Advances in embedding models for text, image, audio, and video drive progress across multiple domains, including retrieval-augmented generation, recommendation systems, vehicle/person reidentification, and face recognition. Many applications in these domains require an efficient method to retrieve items that are close to a given query in the embedding space while satisfying a filter condition based on the item's attributes, a problem known as Filtered Approximate Nearest Neighbor Search (FANNS). In this work, we present a comprehensive survey and taxonomy of FANNS methods and analyze how they are benchmarked in the literature. By doing so, we identify a key challenge in the current FANNS landscape: the lack of diverse and realistic datasets, particularly ones derived from the latest transformer-based text embedding models. To address this, we introduce a novel dataset consisting of embedding vectors for the abstracts of over 2.7 million research articles from the arXiv repository, accompanied by 11 real-world attributes such as authors and categories. We benchmark a wide range of FANNS methods on our novel dataset and find that each method has distinct strengths and limitations; no single approach performs best across all scenarios. ACORN, for example, supports various filter types and performs reliably across dataset scales but is often outperformed by more specialized methods. SeRF shows excellent performance for range filtering on ordered attributes but cannot handle categorical attributes. Filtered-DiskANN and UNG excel on the medium-scale dataset but fail on the large-scale dataset, highlighting the challenge posed by transformer-based embeddings, which are often more than an order of magnitude larger than earlier embeddings. We conclude that no universally best method exists. 

---
# Who's important? -- SUnSET: Synergistic Understanding of Stakeholder, Events and Time for Timeline Generation 

**Authors**: Tiviatis Sim, Kaiwen Yang, Shen Xin, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21903)  

**Abstract**: As news reporting becomes increasingly global and decentralized online, tracking related events across multiple sources presents significant challenges. Existing news summarization methods typically utilizes Large Language Models and Graphical methods on article-based summaries. However, this is not effective since it only considers the textual content of similarly dated articles to understand the gist of the event. To counteract the lack of analysis on the parties involved, it is essential to come up with a novel framework to gauge the importance of stakeholders and the connection of related events through the relevant entities involved. Therefore, we present SUnSET: Synergistic Understanding of Stakeholder, Events and Time for the task of Timeline Summarization (TLS). We leverage powerful Large Language Models (LLMs) to build SET triplets and introduced the use of stakeholder-based ranking to construct a $Relevancy$ metric, which can be extended into general situations. Our experimental results outperform all prior baselines and emerged as the new State-of-the-Art, highlighting the impact of stakeholder information within news article. 

---
# Conversations over Clicks: Impact of Chatbots on Information Search in Interdisciplinary Learning 

**Authors**: Hannah Kim, Sergei L. Kosakovsky Pond, Stephen MacNeil  

**Link**: [PDF](https://arxiv.org/pdf/2507.21490)  

**Abstract**: This full research paper investigates the impact of generative AI (GenAI) on the learner experience, with a focus on how learners engage with and utilize the information it provides. In e-learning environments, learners often need to navigate a complex information space on their own. This challenge is further compounded in interdisciplinary fields like bioinformatics, due to the varied prior knowledge and backgrounds. In this paper, we studied how GenAI influences information search in bioinformatics research: (1) How do interactions with a GenAI chatbot influence learner orienteering behaviors?; and (2) How do learners identify information scent in GenAI chatbot responses? We adopted an autoethnographic approach to investigate these questions. GenAI was found to support orienteering once a learning plan was established, but it was counterproductive prior to that. Moreover, traditionally value-rich information sources such as bullet points and related terms proved less effective when applied to GenAI responses. Information scents were primarily recognized through the presence or absence of prior knowledge of the domain. These findings suggest that GenAI should be adopted into e-learning environments with caution, particularly in interdisciplinary learning contexts. 

---
# Hebbian Memory-Augmented Recurrent Networks: Engram Neurons in Deep Learning 

**Authors**: Daniel Szelogowski  

**Link**: [PDF](https://arxiv.org/pdf/2507.21474)  

**Abstract**: Despite success across diverse tasks, current artificial recurrent network architectures rely primarily on implicit hidden-state memories, limiting their interpretability and ability to model long-range dependencies. In contrast, biological neural systems employ explicit, associative memory traces (i.e., engrams) strengthened through Hebbian synaptic plasticity and activated sparsely during recall. Motivated by these neurobiological insights, we introduce the Engram Neural Network (ENN), a novel recurrent architecture incorporating an explicit, differentiable memory matrix with Hebbian plasticity and sparse, attention-driven retrieval mechanisms. The ENN explicitly models memory formation and recall through dynamic Hebbian traces, improving transparency and interpretability compared to conventional RNN variants. We evaluate the ENN architecture on three canonical benchmarks: MNIST digit classification, CIFAR-10 image sequence modeling, and WikiText-103 language modeling. Our empirical results demonstrate that the ENN achieves accuracy and generalization performance broadly comparable to classical RNN, GRU, and LSTM architectures, with all models converging to similar accuracy and perplexity on the large-scale WikiText-103 task. At the same time, the ENN offers significant enhancements in interpretability through observable memory dynamics. Hebbian trace visualizations further reveal biologically plausible, structured memory formation processes, validating the potential of neuroscience-inspired mechanisms to inform the development of more interpretable and robust deep learning models. 

---
# StructText: A Synthetic Table-to-Text Approach for Benchmark Generation with Multi-Dimensional Evaluation 

**Authors**: Satyananda Kashyap, Sola Shirai, Nandana Mihindukulasooriya, Horst Samulowitz  

**Link**: [PDF](https://arxiv.org/pdf/2507.21340)  

**Abstract**: Extracting structured information from text, such as key-value pairs that could augment tabular data, is quite useful in many enterprise use cases. Although large language models (LLMs) have enabled numerous automated pipelines for converting natural language into structured formats, there is still a lack of benchmarks for evaluating their extraction quality, especially in specific domains or focused documents specific to a given organization. Building such benchmarks by manual annotations is labour-intensive and limits the size and scalability of the benchmarks. In this work, we present StructText, an end-to-end framework for automatically generating high-fidelity benchmarks for key-value extraction from text using existing tabular data. It uses available tabular data as structured ground truth, and follows a two-stage ``plan-then-execute'' pipeline to synthetically generate corresponding natural-language text. To ensure alignment between text and structured source, we introduce a multi-dimensional evaluation strategy that combines (a) LLM-based judgments on factuality, hallucination, and coherence and (b) objective extraction metrics measuring numeric and temporal accuracy. We evaluated the proposed method on 71,539 examples across 49 datasets. Results reveal that while LLMs achieve strong factual accuracy and avoid hallucination, they struggle with narrative coherence in producing extractable text. Notably, models presume numerical and temporal information with high fidelity yet this information becomes embedded in narratives that resist automated extraction. We release a framework, including datasets, evaluation tools, and baseline extraction systems, to support continued research. 

---
# SemRAG: Semantic Knowledge-Augmented RAG for Improved Question-Answering 

**Authors**: Kezhen Zhong, Basem Suleiman, Abdelkarim Erradi, Shijing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21110)  

**Abstract**: This paper introduces SemRAG, an enhanced Retrieval Augmented Generation (RAG) framework that efficiently integrates domain-specific knowledge using semantic chunking and knowledge graphs without extensive fine-tuning. Integrating domain-specific knowledge into large language models (LLMs) is crucial for improving their performance in specialized tasks. Yet, existing adaptations are computationally expensive, prone to overfitting and limit scalability. To address these challenges, SemRAG employs a semantic chunking algorithm that segments documents based on the cosine similarity from sentence embeddings, preserving semantic coherence while reducing computational overhead. Additionally, by structuring retrieved information into knowledge graphs, SemRAG captures relationships between entities, improving retrieval accuracy and contextual understanding. Experimental results on MultiHop RAG and Wikipedia datasets demonstrate SemRAG has significantly enhances the relevance and correctness of retrieved information from the Knowledge Graph, outperforming traditional RAG methods. Furthermore, we investigate the optimization of buffer sizes for different data corpus, as optimizing buffer sizes tailored to specific datasets can further improve retrieval performance, as integration of knowledge graphs strengthens entity relationships for better contextual comprehension. The primary advantage of SemRAG is its ability to create an efficient, accurate domain-specific LLM pipeline while avoiding resource-intensive fine-tuning. This makes it a practical and scalable approach aligned with sustainability goals, offering a viable solution for AI applications in domain-specific fields. 

---
