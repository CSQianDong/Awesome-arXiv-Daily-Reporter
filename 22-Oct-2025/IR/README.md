# LLMs as Sparse Retrievers:A Framework for First-Stage Product Search 

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Sen Li, Wenjun Peng, Fuyu Lv, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.18527)  

**Abstract**: Product search is a crucial component of modern e-commerce platforms, with billions of user queries every day. In product search systems, first-stage retrieval should achieve high recall while ensuring efficient online deployment. Sparse retrieval is particularly attractive in this context due to its interpretability and storage efficiency. However, sparse retrieval methods suffer from severe vocabulary mismatch issues, leading to suboptimal performance in product search this http URL their potential for semantic analysis, large language models (LLMs) offer a promising avenue for mitigating vocabulary mismatch issues and thereby improving retrieval quality. Directly applying LLMs to sparse retrieval in product search exposes two key challenges:(1)Queries and product titles are typically short and highly susceptible to LLM-induced hallucinations, such as generating irrelevant expansion terms or underweighting critical literal terms like brand names and model numbers;(2)The large vocabulary space of LLMs leads to difficulty in initializing training effectively, making it challenging to learn meaningful sparse representations in such ultra-high-dimensional this http URL address these challenges, we propose PROSPER, a framework for PROduct search leveraging LLMs as SParsE Retrievers. PROSPER incorporates: (1)A literal residual network that alleviates hallucination in lexical expansion by reinforcing underweighted literal terms through a residual compensation mechanism; and (2)A lexical focusing window that facilitates effective training initialization via a coarse-to-fine sparsification this http URL offline and online experiments show that PROSPER significantly outperforms sparse baselines and achieves recall performance comparable to advanced dense retrievers, while also achieving revenue increments online. 

---
# Evaluating LLM-Based Mobile App Recommendations: An Empirical Study 

**Authors**: Quim Motger, Xavier Franch, Vincenzo Gervasi, Jordi Marco  

**Link**: [PDF](https://arxiv.org/pdf/2510.18364)  

**Abstract**: Large Language Models (LLMs) are increasingly used to recommend mobile applications through natural language prompts, offering a flexible alternative to keyword-based app store search. Yet, the reasoning behind these recommendations remains opaque, raising questions about their consistency, explainability, and alignment with traditional App Store Optimization (ASO) metrics. In this paper, we present an empirical analysis of how widely-used general purpose LLMs generate, justify, and rank mobile app recommendations. Our contributions are: (i) a taxonomy of 16 generalizable ranking criteria elicited from LLM outputs; (ii) a systematic evaluation framework to analyse recommendation consistency and responsiveness to explicit ranking instructions; and (iii) a replication package to support reproducibility and future research on AI-based recommendation systems. Our findings reveal that LLMs rely on a broad yet fragmented set of ranking criteria, only partially aligned with standard ASO metrics. While top-ranked apps tend to be consistent across runs, variability increases with ranking depth and search specificity. LLMs exhibit varying sensitivity to explicit ranking instructions - ranging from substantial adaptations to near-identical outputs - highlighting their complex reasoning dynamics in conversational app discovery. Our results aim to support end-users, app developers, and recommender-systems researchers in navigating the emerging landscape of conversational app discovery. 

---
# Enhancing Hotel Recommendations with AI: LLM-Based Review Summarization and Query-Driven Insights 

**Authors**: Nikolaos Belibasakis, Anastasios Giannaros, Ioanna Giannoukou, Spyros Sioutas  

**Link**: [PDF](https://arxiv.org/pdf/2510.18277)  

**Abstract**: The increasing number of data a booking platform such as this http URL and AirBnB offers make it challenging for interested parties to browse through the available accommodations and analyze reviews in an efficient way. Efforts have been made from the booking platform providers to utilize recommender systems in an effort to enable the user to filter the results by factors such as stars, amenities, cost but most valuable insights can be provided by the unstructured text-based reviews. Going through these reviews one-by-one requires a substantial amount of time to be devoted while a respectable percentage of the reviews won't provide to the user what they are actually looking for.
This research publication explores how Large Language Models (LLMs) can enhance short rental apartments recommendations by summarizing and mining key insights from user reviews. The web application presented in this paper, named "instaGuide", automates the procedure of isolating the text-based user reviews from a property on the this http URL platform, synthesizing the summary of the reviews, and enabling the user to query specific aspects of the property in an effort to gain feedback on their personal questions/criteria.
During the development of the instaGuide tool, numerous LLM models were evaluated based on accuracy, cost, and response quality. The results suggest that the LLM-powered summarization reduces significantly the amount of time the users need to devote on their search for the right short rental apartment, improving the overall decision-making procedure. 

---
# LIME: Link-based user-item Interaction Modeling with decoupled xor attention for Efficient test time scaling 

**Authors**: Yunjiang Jiang, Ayush Agarwal, Yang Liu, Bi Xue  

**Link**: [PDF](https://arxiv.org/pdf/2510.18239)  

**Abstract**: Scaling large recommendation systems requires advancing three major frontiers: processing longer user histories, expanding candidate sets, and increasing model capacity. While promising, transformers' computational cost scales quadratically with the user sequence length and linearly with the number of candidates. This trade-off makes it prohibitively expensive to expand candidate sets or increase sequence length at inference, despite the significant performance improvements.
We introduce \textbf{LIME}, a novel architecture that resolves this trade-off. Through two key innovations, LIME fundamentally reduces computational complexity. First, low-rank ``link embeddings" enable pre-computation of attention weights by decoupling user and candidate interactions, making the inference cost nearly independent of candidate set size. Second, a linear attention mechanism, \textbf{LIME-XOR}, reduces the complexity with respect to user sequence length from quadratic ($O(N^2)$) to linear ($O(N)$).
Experiments on public and industrial datasets show LIME achieves near-parity with state-of-the-art transformers but with a 10$\times$ inference speedup on large candidate sets or long sequence lengths. When tested on a major recommendation platform, LIME improved user engagement while maintaining minimal inference costs with respect to candidate set size and user history length, establishing a new paradigm for efficient and expressive recommendation systems. 

---
# From AutoRecSys to AutoRecLab: A Call to Build, Evaluate, and Govern Autonomous Recommender-Systems Research Labs 

**Authors**: Joeran Beel, Bela Gipp, Tobias Vente, Moritz Baumgart, Philipp Meister  

**Link**: [PDF](https://arxiv.org/pdf/2510.18104)  

**Abstract**: Recommender-systems research has accelerated model and evaluation advances, yet largely neglects automating the research process itself. We argue for a shift from narrow AutoRecSys tools -- focused on algorithm selection and hyper-parameter tuning -- to an Autonomous Recommender-Systems Research Lab (AutoRecLab) that integrates end-to-end automation: problem ideation, literature analysis, experimental design and execution, result interpretation, manuscript drafting, and provenance logging. Drawing on recent progress in automated science (e.g., multi-agent AI Scientist and AI Co-Scientist systems), we outline an agenda for the RecSys community: (1) build open AutoRecLab prototypes that combine LLM-driven ideation and reporting with automated experimentation; (2) establish benchmarks and competitions that evaluate agents on producing reproducible RecSys findings with minimal human input; (3) create review venues for transparently AI-generated submissions; (4) define standards for attribution and reproducibility via detailed research logs and metadata; and (5) foster interdisciplinary dialogue on ethics, governance, privacy, and fairness in autonomous research. Advancing this agenda can increase research throughput, surface non-obvious insights, and position RecSys to contribute to emerging Artificial Research Intelligence. We conclude with a call to organise a community retreat to coordinate next steps and co-author guidance for the responsible integration of automated research systems. 

---
# ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization 

**Authors**: Yuanhe Guo, Linxi Xie, Zhuoran Chen, Kangrui Yu, Ryan Po, Guandao Yang, Gordon Wetztein, Hongyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2510.18433)  

**Abstract**: We introduce ImageGem, a dataset for studying generative models that understand fine-grained individual preferences. We posit that a key challenge hindering the development of such a generative model is the lack of in-the-wild and fine-grained user preference annotations. Our dataset features real-world interaction data from 57K users, who collectively have built 242K customized LoRAs, written 3M text prompts, and created 5M generated images. With user preference annotations from our dataset, we were able to train better preference alignment models. In addition, leveraging individual user preference, we investigated the performance of retrieval models and a vision-language model on personalized image retrieval and generative model recommendation. Finally, we propose an end-to-end framework for editing customized diffusion models in a latent weight space to align with individual user preferences. Our results demonstrate that the ImageGem dataset enables, for the first time, a new paradigm for generative model personalization. 

---
# Censorship Chokepoints: New Battlegrounds for Regional Surveillance, Censorship and Influence on the Internet 

**Authors**: Yong Zhang, Nishanth Sastry  

**Link**: [PDF](https://arxiv.org/pdf/2510.18394)  

**Abstract**: Undoubtedly, the Internet has become one of the most important conduits to information for the general public. Nonetheless, Internet access can be and has been limited systematically or blocked completely during political events in numerous countries and regions by various censorship mechanisms. Depending on where the core filtering component is situated, censorship techniques have been classified as client-based, server-based, or network-based. However, as the Internet evolves rapidly, new and sophisticated censorship techniques have emerged, which involve techniques that cut across locations and involve new forms of hurdles to information access. We argue that modern censorship can be better understood through a new lens that we term chokepoints, which identifies bottlenecks in the content production or delivery cycle where efficient new forms of large-scale client-side surveillance and filtering mechanisms have emerged. 

---
# KrishokBondhu: A Retrieval-Augmented Voice-Based Agricultural Advisory Call Center for Bengali Farmers 

**Authors**: Mohd Ruhul Ameen, Akif Islam, Farjana Aktar, M. Saifuzzaman Rafat  

**Link**: [PDF](https://arxiv.org/pdf/2510.18355)  

**Abstract**: In Bangladesh, many farmers continue to face challenges in accessing timely, expert-level agricultural guidance. This paper presents KrishokBondhu, a voice-enabled, call-centre-integrated advisory platform built on a Retrieval-Augmented Generation (RAG) framework, designed specifically for Bengali-speaking farmers. The system aggregates authoritative agricultural handbooks, extension manuals, and NGO publications; applies Optical Character Recognition (OCR) and document-parsing pipelines to digitize and structure the content; and indexes this corpus in a vector database for efficient semantic retrieval. Through a simple phone-based interface, farmers can call the system to receive real-time, context-aware advice: speech-to-text converts the Bengali query, the RAG module retrieves relevant content, a large language model (Gemma 3-4B) generates a context-grounded response, and text-to-speech delivers the answer in natural spoken Bengali. In a pilot evaluation, KrishokBondhu produced high-quality responses for 72.7% of diverse agricultural queries covering crop management, disease control, and cultivation practices. Compared to the KisanQRS benchmark, the system achieved a composite score of 4.53 (vs. 3.13) on a 5-point scale, a 44.7% improvement, with especially large gains in contextual richness (+367%) and completeness (+100.4%), while maintaining comparable relevance and technical specificity. Semantic similarity analysis further revealed a strong correlation between retrieved context and answer quality, emphasizing the importance of grounding generative responses in curated documentation. KrishokBondhu demonstrates the feasibility of integrating call-centre accessibility, multilingual voice interaction, and modern RAG techniques to deliver expert-level agricultural guidance to remote Bangladeshi farmers, paving the way toward a fully AI-driven agricultural advisory ecosystem. 

---
