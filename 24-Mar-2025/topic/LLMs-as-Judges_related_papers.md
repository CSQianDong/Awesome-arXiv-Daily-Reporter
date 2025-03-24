# Automating Adjudication of Cardiovascular Events Using Large Language Models 

**Authors**: Sonish Sivarajkumar, Kimia Ameri, Chuqin Li, Yanshan Wang, Min Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17222)  

**Abstract**: Cardiovascular events, such as heart attacks and strokes, remain a leading cause of mortality globally, necessitating meticulous monitoring and adjudication in clinical trials. This process, traditionally performed manually by clinical experts, is time-consuming, resource-intensive, and prone to inter-reviewer variability, potentially introducing bias and hindering trial progress. This study addresses these critical limitations by presenting a novel framework for automating the adjudication of cardiovascular events in clinical trials using Large Language Models (LLMs). We developed a two-stage approach: first, employing an LLM-based pipeline for event information extraction from unstructured clinical data and second, using an LLM-based adjudication process guided by a Tree of Thoughts approach and clinical endpoint committee (CEC) guidelines. Using cardiovascular event-specific clinical trial data, the framework achieved an F1-score of 0.82 for event extraction and an accuracy of 0.68 for adjudication. Furthermore, we introduce the CLEART score, a novel, automated metric specifically designed for evaluating the quality of AI-generated clinical reasoning in adjudicating cardiovascular events. This approach demonstrates significant potential for substantially reducing adjudication time and costs while maintaining high-quality, consistent, and auditable outcomes in clinical trials. The reduced variability and enhanced standardization also allow for faster identification and mitigation of risks associated with cardiovascular therapies. 

---
# Summarization Metrics for Spanish and Basque: Do Automatic Scores and LLM-Judges Correlate with Humans? 

**Authors**: Jeremy Barnes, Naiara Perez, Alba Bonet-Jover, Begoña Altuna  

**Link**: [PDF](https://arxiv.org/pdf/2503.17039)  

**Abstract**: Studies on evaluation metrics and LLM-as-a-Judge models for automatic text summarization have largely been focused on English, limiting our understanding of their effectiveness in other languages. Through our new dataset BASSE (BAsque and Spanish Summarization Evaluation), we address this situation by collecting human judgments on 2,040 abstractive summaries in Basque and Spanish, generated either manually or by five LLMs with four different prompts. For each summary, annotators evaluated five criteria on a 5-point Likert scale: coherence, consistency, fluency, relevance, and 5W1H. We use these data to reevaluate traditional automatic metrics used for evaluating summaries, as well as several LLM-as-a-Judge models that show strong performance on this task in English. Our results show that currently proprietary judge LLMs have the highest correlation with human judgments, followed by criteria-specific automatic metrics, while open-sourced judge LLMs perform poorly. We release BASSE and our code publicly, along with the first large-scale Basque summarization dataset containing 22,525 news articles with their subheads. 

---
# Scalable Evaluation of Online Moderation Strategies via Synthetic Simulations 

**Authors**: Dimitris Tsirmpas, Ion Androutsopoulos, John Pavlopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.16505)  

**Abstract**: Despite the ever-growing importance of online moderation, there has been no large-scale study evaluating the effectiveness of alternative moderation strategies. This is largely due to the lack of appropriate datasets, and the difficulty of getting human discussants, moderators, and evaluators involved in multiple experiments. In this paper, we propose a methodology for leveraging synthetic experiments performed exclusively by Large Language Models (LLMs) to initially bypass the need for human participation in experiments involving online moderation. We evaluate six LLM moderation configurations; two currently used real-life moderation strategies (guidelines issued for human moderators for online moderation and real-life facilitation), two baseline strategies (guidelines elicited for LLM alignment work, and LLM moderation with minimal prompting) a baseline with no moderator at all, as well as our own proposed strategy inspired by a Reinforcement Learning (RL) formulation of the problem. We find that our own moderation strategy significantly outperforms established moderation guidelines, as well as out-of-the-box LLM moderation. We also find that smaller LLMs, with less intensive instruction-tuning, can create more varied discussions than larger models. In order to run these experiments, we create and release an efficient, purpose-built, open-source Python framework, dubbed "SynDisco" to easily simulate hundreds of discussions using LLM user-agents and moderators. Additionally, we release the Virtual Moderation Dataset (VMD), a large dataset of LLM-generated and LLM-annotated discussions, generated by three families of open-source LLMs accompanied by an exploratory analysis of the dataset. 

---
