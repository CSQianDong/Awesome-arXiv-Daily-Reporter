# Leveraging Graph Retrieval-Augmented Generation to Support Learners' Understanding of Knowledge Concepts in MOOCs 

**Authors**: Mohamed Abdelmagied, Mohamed Amine Chatti, Shoeb Joarder, Qurat Ul Ain, Rawaa Alatrash  

**Link**: [PDF](https://arxiv.org/pdf/2505.10074)  

**Abstract**: Massive Open Online Courses (MOOCs) lack direct interaction between learners and instructors, making it challenging for learners to understand new knowledge concepts. Recently, learners have increasingly used Large Language Models (LLMs) to support them in acquiring new knowledge. However, LLMs are prone to hallucinations which limits their reliability. Retrieval-Augmented Generation (RAG) addresses this issue by retrieving relevant documents before generating a response. However, the application of RAG across different MOOCs is limited by unstructured learning material. Furthermore, current RAG systems do not actively guide learners toward their learning needs. To address these challenges, we propose a Graph RAG pipeline that leverages Educational Knowledge Graphs (EduKGs) and Personal Knowledge Graphs (PKGs) to guide learners to understand knowledge concepts in the MOOC platform CourseMapper. Specifically, we implement (1) a PKG-based Question Generation method to recommend personalized questions for learners in context, and (2) an EduKG-based Question Answering method that leverages the relationships between knowledge concepts in the EduKG to answer learner selected questions. To evaluate both methods, we conducted a study with 3 expert instructors on 3 different MOOCs in the MOOC platform CourseMapper. The results of the evaluation show the potential of Graph RAG to empower learners to understand new knowledge concepts in a personalized learning experience. 

---
# Personalizing Large Language Models using Retrieval Augmented Generation and Knowledge Graph 

**Authors**: Deeksha Prahlad, Chanhee Lee, Dongha Kim, Hokeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.09945)  

**Abstract**: The advent of large language models (LLMs) has allowed numerous applications, including the generation of queried responses, to be leveraged in chatbots and other conversational assistants. Being trained on a plethora of data, LLMs often undergo high levels of over-fitting, resulting in the generation of extra and incorrect data, thus causing hallucinations in output generation. One of the root causes of such problems is the lack of timely, factual, and personalized information fed to the LLM. In this paper, we propose an approach to address these problems by introducing retrieval augmented generation (RAG) using knowledge graphs (KGs) to assist the LLM in personalized response generation tailored to the users. KGs have the advantage of storing continuously updated factual information in a structured way. While our KGs can be used for a variety of frequently updated personal data, such as calendar, contact, and location data, we focus on calendar data in this paper. Our experimental results show that our approach works significantly better in understanding personal information and generating accurate responses compared to the baseline LLMs using personal data as text inputs, with a moderate reduction in response time. 

---
# CL-RAG: Bridging the Gap in Retrieval-Augmented Generation with Curriculum Learning 

**Authors**: Shaohan Wang, Licheng Zhang, Zheren Fu, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10493)  

**Abstract**: Retrieval-Augmented Generation (RAG) is an effective method to enhance the capabilities of large language models (LLMs). Existing methods focus on optimizing the retriever or generator in the RAG system by directly utilizing the top-k retrieved documents. However, the documents effectiveness are various significantly across user queries, i.e. some documents provide valuable knowledge while others totally lack critical information. It hinders the retriever and generator's adaptation during training. Inspired by human cognitive learning, curriculum learning trains models using samples progressing from easy to difficult, thus enhancing their generalization ability, and we integrate this effective paradigm to the training of the RAG system. In this paper, we propose a multi-stage Curriculum Learning based RAG system training framework, named CL-RAG. We first construct training data with multiple difficulty levels for the retriever and generator separately through sample evolution. Then, we train the model in stages based on the curriculum learning approach, thereby optimizing the overall performance and generalization of the RAG system more effectively. Our CL-RAG framework demonstrates consistent effectiveness across four open-domain QA datasets, achieving performance gains of 2% to 4% over multiple advanced methods. 

---
# GE-Chat: A Graph Enhanced RAG Framework for Evidential Response Generation of LLMs 

**Authors**: Longchao Da, Parth Mitesh Shah, Kuan-Ru Liou, Jiaxing Zhang, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.10143)  

**Abstract**: Large Language Models are now key assistants in human decision-making processes. However, a common note always seems to follow: "LLMs can make mistakes. Be careful with important info." This points to the reality that not all outputs from LLMs are dependable, and users must evaluate them manually. The challenge deepens as hallucinated responses, often presented with seemingly plausible explanations, create complications and raise trust issues among users. To tackle such issue, this paper proposes GE-Chat, a knowledge Graph enhanced retrieval-augmented generation framework to provide Evidence-based response generation. Specifically, when the user uploads a material document, a knowledge graph will be created, which helps construct a retrieval-augmented agent, enhancing the agent's responses with additional knowledge beyond its training corpus. Then we leverage Chain-of-Thought (CoT) logic generation, n-hop sub-graph searching, and entailment-based sentence generation to realize accurate evidence retrieval. We demonstrate that our method improves the existing models' performance in terms of identifying the exact evidence in a free-form context, providing a reliable way to examine the resources of LLM's conclusion and help with the judgment of the trustworthiness. 

---
# XRAG: Cross-lingual Retrieval-Augmented Generation 

**Authors**: Wei Liu, Sony Trenous, Leonardo F. R. Ribeiro, Bill Byrne, Felix Hieber  

**Link**: [PDF](https://arxiv.org/pdf/2505.10089)  

**Abstract**: We propose XRAG, a novel benchmark designed to evaluate the generation abilities of LLMs in cross-lingual Retrieval-Augmented Generation (RAG) settings where the user language does not match the retrieval results. XRAG is constructed from recent news articles to ensure that its questions require external knowledge to be answered. It covers the real-world scenarios of monolingual and multilingual retrieval, and provides relevancy annotations for each retrieved document. Our novel dataset construction pipeline results in questions that require complex reasoning, as evidenced by the significant gap between human and LLM performance. Consequently, XRAG serves as a valuable benchmark for studying LLM reasoning abilities, even before considering the additional cross-lingual complexity. Experimental results on five LLMs uncover two previously unreported challenges in cross-lingual RAG: 1) in the monolingual retrieval setting, all evaluated models struggle with response language correctness; 2) in the multilingual retrieval setting, the main challenge lies in reasoning over retrieved information across languages rather than generation of non-English text. 

---
# Do Large Language Models Know Conflict? Investigating Parametric vs. Non-Parametric Knowledge of LLMs for Conflict Forecasting 

**Authors**: Apollinaire Poli Nemkova, Sarath Chandra Lingareddy, Sagnik Ray Choudhury, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2505.09852)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance across natural language tasks, but their ability to forecast violent conflict remains underexplored. We investigate whether LLMs possess meaningful parametric knowledge-encoded in their pretrained weights-to predict conflict escalation and fatalities without external data. This is critical for early warning systems, humanitarian planning, and policy-making. We compare this parametric knowledge with non-parametric capabilities, where LLMs access structured and unstructured context from conflict datasets (e.g., ACLED, GDELT) and recent news reports via Retrieval-Augmented Generation (RAG). Incorporating external information could enhance model performance by providing up-to-date context otherwise missing from pretrained weights. Our two-part evaluation framework spans 2020-2024 across conflict-prone regions in the Horn of Africa and the Middle East. In the parametric setting, LLMs predict conflict trends and fatalities relying only on pretrained knowledge. In the non-parametric setting, models receive summaries of recent conflict events, indicators, and geopolitical developments. We compare predicted conflict trend labels (e.g., Escalate, Stable Conflict, De-escalate, Peace) and fatalities against historical data. Our findings highlight the strengths and limitations of LLMs for conflict forecasting and the benefits of augmenting them with structured external knowledge. 

---
