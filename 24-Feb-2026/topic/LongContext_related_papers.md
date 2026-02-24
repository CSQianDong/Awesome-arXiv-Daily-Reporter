# Learning to Reason for Multi-Step Retrieval of Personal Context in Personalized Question Answering 

**Authors**: Maryam Amirizaniani, Alireza Salemi, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2602.19317)  

**Abstract**: Personalization in Question Answering (QA) requires answers that are both accurate and aligned with users' background, preferences, and historical context. Existing state-of-the-art methods primarily rely on retrieval-augmented generation (RAG) solutions that construct personal context by retrieving relevant items from the user's profile. Existing methods use the user's query directly to retrieve personal documents, and such strategies often lead to surface-level personalization. We propose PR2 (Personalized Retrieval-Augmented Reasoning), a reinforcement learning framework that integrates reasoning and retrieval from personal context for personalization. PR2 learns adaptive retrieval-reasoning policies, determining when to retrieve, what evidence to retrieve from user profiles, and how to incorporate it into intermediate reasoning steps. By optimizing multi-turn reasoning trajectories under a personalized reward function, the framework reinforces reasoning paths that better align with user-specific preferences and contextual signals reflected by the reward model. Extensive experiments on the LaMP-QA benchmark using three LLMs show that PR2 consistently outperforms strong baselines, achieving an average relative improvement of 8.8%-12% in personalized QA. 

---
# Depth-Structured Music Recurrence: Budgeted Recurrent Attention for Full-Piece Symbolic Music Modeling 

**Authors**: Yungang Yi  

**Link**: [PDF](https://arxiv.org/pdf/2602.19816)  

**Abstract**: Long-context modeling is essential for symbolic music generation, since motif repetition and developmental variation can span thousands of musical events. However, practical composition and performance workflows frequently rely on resource-limited devices (e.g., electronic instruments and portable computers), making heavy memory and attention computation difficult to deploy. We introduce Depth-Structured Music Recurrence (DSMR), a recurrent long-context Transformer for full-piece symbolic music modeling that extends context beyond fixed-length excerpts via segment-level recurrence with detached cross-segment states, featuring a layer-wise memory-horizon schedule that budgets recurrent KV states across depth. DSMR is trained in a single left-to-right pass over each complete composition, akin to how a musician experiences it from beginning to end, while carrying recurrent cross-segment states forward. Within this recurrent framework, we systematically study how depth-wise horizon allocations affect optimization, best-checkpoint perplexity, and efficiency. By allocating different history-window lengths across layers while keeping the total recurrent-state budget fixed, DSMR creates depth-dependent temporal receptive fields within a recurrent attention stack without reducing compute depth. Our main instantiation is a two-scale DSMR schedule that allocates long history windows to lower layers and a uniform short window to the remaining layers. Experiments on the piano performance dataset MAESTRO demonstrate that two-scale DSMR provides a practical quality--efficiency recipe for full-length long-context symbolic music modeling with recurrent attention under limited computational resources. 

---
# Learning to Remember: End-to-End Training of Memory Agents for Long-Context Reasoning 

**Authors**: Kehao Zhang, Shangtong Gui, Sheng Yang, Wei Chen, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2602.18493)  

**Abstract**: Long-context LLMs and Retrieval-Augmented Generation (RAG) systems process information passively, deferring state tracking, contradiction resolution, and evidence aggregation to query time, which becomes brittle under ultra long streams with frequent updates. We propose the Unified Memory Agent (UMA), an end-to-end reinforcement learning framework that unifies memory operations and question answering within a single policy. UMA maintains a dual memory representation: a compact core summary for global context and a structured Memory Bank that supports explicit CRUD (create, update, delete, reorganize) over key value entries, enabling proactive consolidation during streaming. To evaluate long-horizon memory behavior, we introduce Ledger-QA, a diagnostic benchmark for continuous state tracking where answers are latent values derived from accumulated updates rather than lo cal span retrieval. Across 13 datasets spanning Ledger-QA, Test-Time Learning, and Accurate Retrieval, UMA substantially outperforms long-context and RAG baselines on dynamic reasoning and learning tasks while remaining competitive on standard retrieval benchmarks, underscoring the importance of learned, end-to-end memory management. 

---
