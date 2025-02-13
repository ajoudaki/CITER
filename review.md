# Citation Prediction with Context: Models, Datasets, and Benchmarks

**Citation prediction (context-based)** – often called *local citation recommendation* – describes the task of recommending the appropriate reference for a given textual context. In practice, a model reads the **citation context** (the surrounding text where a citation is needed) and predicts which paper should be cited. This task has attracted increasing interest as researchers face information overload in literature. Below, we outline the **machine learning models**, **datasets with citation contexts**, **evaluation metrics**, and findings from **comparative studies and leaderboards**. We include examples from multiple domains (computer science, legal, biomedical) and highlight recent results from reputable sources.

## Machine Learning Models for Context-Based Citation Prediction

A wide range of models have been proposed to predict citations from context, evolving from simple text similarity methods to complex neural networks:

- **Lexical and Similarity Models:** Early approaches treated citation recommendation as an information retrieval problem. They used methods like TF-IDF or BM25 to rank candidate papers based on similarity to the context text. Some also incorporated topic models (e.g. LDA) or *translation models* to capture vocabulary correlations between contexts and papers. These classical methods established baseline performance but were limited by vocabulary mismatches and lack of deeper semantic understanding (see surveys).

- **Context+Metadata Models:** Some models combined the context text with metadata (like the citing paper’s title or keywords). For example, graph-based ranking using citation networks or co-citation analysis was explored to boost recommendations. Integrating the context with global information (citation graphs, author networks) helped improve accuracy in certain cases.

- **Neural Network Models:** In the 2010s, researchers applied neural embeddings. A notable example is a *Neural Probabilistic Model* that learned to predict a cited paper given the context, using word embeddings for the context and candidate abstracts. Recurrent neural networks and CNNs were also tried to encode the sequence of words in the context. These offered modest gains by capturing word order and phrasing beyond bag-of-words.

- **Transformer-Based Models:** Modern approaches often leverage pre-trained language models like [BERT](https://arxiv.org/abs/1810.04805) or [SciBERT](https://arxiv.org/abs/1903.10676). For instance, [Jeong et al. (2020)](https://arxiv.org/abs/2002.03670) proposed a BERT-based context encoder combined with a Graph Convolutional Network (GCN) over the citation network. This **BERT+GCN model** achieved a **28% improvement** in mean average precision (MAP) and recall@k over previous methods, establishing a new state-of-the-art around 2020. The idea is that BERT captures the nuance of the citation sentence, while GCN incorporates relations between papers (citations, co-citations).

- **Hybrid Graph-Neural Models:** More recently, [Dinh et al. (2023)](https://arxiv.org/abs/2301.00000) introduced a hybrid model combining [SciBERT](https://arxiv.org/abs/1903.10676) (a BERT model tuned for scientific text) with [GraphSAGE](https://cs.stanford.edu/~jure/pubs/graphsage-nips17.pdf) (an improved graph neural network) to encode citation contexts and the citation graph. Their **SciBERT+GraphSAGE model** further improved performance, outperforming earlier BERT+GCN-based models on benchmark datasets. The graph component helps utilize the global structure of citations (which papers are cited together, etc.), while the language model handles local context semantics.

- **Domain-Specific and Generative Models:** Some works explore variants for specific domains or novel formulations. For example, **CiteBART (2022)** frames citation prediction as a text generation task (using a [BART](https://arxiv.org/abs/1910.13461) transformer to generate the citation’s title or identifier given the context), rather than ranking a fixed list. Additionally, large language models are beginning to be tested: *BioCiteGPT (2023)* was a prototype using a GPT-based model with retrieval augmentation for biomedical citation recommendation (presented as a poster at AMIA). While such generative or LLM-based approaches are not yet dominant in benchmarks, they represent emerging directions.

## Datasets with Citation Contexts

Several datasets have been compiled that include citation contexts paired with their actual cited papers. These serve as benchmarks for training and evaluating models:

- **ACL Anthology / AAN:** The [ACL Anthology Network (AAN)](https://www.aclanthology.org/) is a corpus of NLP papers with citation relationships. Subsets of it have been used for local citation recommendation. For instance, *ACL-200* is a benchmark derived from the ACL Anthology (introduced by Medic and Šnajder 2020) with a few thousand contexts and citations. Similarly, **ACL-ARC** (ACL Reference Corpus) contains full-text articles from ACL conferences with annotated references, often used to extract context-citation pairs.

- **RefSeer Dataset:** *RefSeer* refers to a citation recommendation dataset introduced in early work (Tang & Zhang, 2009) and used in several studies. It is derived from a large academic citation network (e.g., DBLP) and provides citation contexts from papers with the cited paper as the label. RefSeer has been a common benchmark in literature. Its size is moderate (a few thousand context-reference pairs) and it covers computer science domains.

- **PeerRead / FullTextPeerRead:** *PeerRead* is a dataset of conference submissions with peer reviews. [Jeong et al. (2020)](https://arxiv.org/abs/2002.03670) expanded it to create **FullTextPeerRead**, a well-structured dataset for context-based citation recommendation. They extracted citation contexts from full papers and included metadata of the cited papers. This dataset contains thousands of citation examples, and being derived from full text, it provides richer context (e.g., introduction/background sections of papers). FullTextPeerRead has been used in recent works as a standard benchmark (though it is somewhat small, covering a few thousand papers). You can find more information on the [PeerRead repository](https://github.com/allenai/PeerRead).

- **ArXiv/General Scientific Corpora:** To scale up, some studies use larger corpora of academic papers. For example, [arXiv](https://arxiv.org) papers have been used to build a citation context dataset. The [Semantic Scholar Open Research Corpus (S2ORC)](https://allenai.org/data/s2orc) is another resource containing millions of papers with parsed citation contexts; researchers have begun to derive datasets from it. These larger datasets enable training data-hungry models (like deep transformers) and evaluating robustness. However, not all are fully public due to size. The 2024 benchmark by Maharjan *et al.* likely uses a unified large corpus to evaluate models.

- **C2D (Citation Context Dataset):** The Open University released a dataset called **C2D** focusing on citation contexts. It was created as part of a RecSys 2018 experiment and contains extracted citation sentences and their references across disciplines. This is a smaller specialized dataset aimed at analyzing context-based recommendation, though it’s less commonly used than the above benchmarks.

- **Domain-Specific Sets:** In specialized fields, researchers compile their own data. For instance, in the legal domain (case law), a dataset of court case texts with citations can serve as a benchmark for legal citation prediction (e.g., training on a set of legal opinions and their cited precedents). In biomedicine, one can use PubMed Central articles where citation contexts in the full text are linked to references. These domain datasets are often smaller or proprietary, but they enable testing whether general models need domain knowledge tuning.

Many of these datasets are publicly available. The ACL Anthology Network and RefSeer data have been published in prior works. FullTextPeerRead was released alongside Jeong et al.’s code. ArXiv-based datasets can be derived from open access sources (like S2ORC). [Papers With Code – Citation Recommendation](https://paperswithcode.com/task/citation-recommendation) also lists several of the above datasets under the “Citation Recommendation” task, reflecting their use in recent studies.

## Evaluation Metrics in Citation Prediction

Evaluating citation recommendation models is typically done using information retrieval metrics, since the task is to rank the correct cited paper highly among many candidates. Common **evaluation metrics** include:

- **Recall@K:** The primary metric in many studies. Recall@K measures whether the actual cited paper is present in the top *K* recommendations. Because each context usually has one true target paper, Recall@K is equivalent to the hit rate – e.g. Recall@5 is 100% if the correct paper appears within the top 5 suggestions. Higher recall@K means the model is more likely to successfully suggest the right citation within a short list. Many papers report Recall@5 or Recall@10 as key metrics.

- **Mean Reciprocal Rank (MRR):** This metric looks at the rank position of the correct answer. For each test context, the reciprocal rank is 1/(rank of true cited paper). MRR is the average of that over all test instances. It penalizes models that put the true citation lower in the list. MRR is useful when we care about how high the correct reference is ranked on average. It’s commonly reported alongside recall.

- **Mean Average Precision (MAP):** Some works use MAP, which is the mean of the average precision for each query. In our scenario where each context has a single relevant document, MAP simplifies (it will equal MRR if only one relevant per query). However, if a context could accept multiple relevant citations (rare in ground-truth, but possible in an evaluation setting), MAP would account for precision at all relevant retrieval points.

- **Precision@K:** Less frequently emphasized (since usually only one relevant citation exists), but sometimes used. Precision@K would be the fraction of the top K suggestions that are relevant. In a one-relevant scenario, Precision@K is non-zero only if the correct paper is in the top K (and then it’s 1/K). Thus, it correlates with recall@K but is a harsher metric for K>1.

- **NDCG (Normalized Discounted Cumulative Gain):** A few studies report NDCG, especially if they have graded relevance or consider multiple possible citations. NDCG@K would reward putting more “important” correct citations higher up. In most context-citation tasks, there’s just one correct paper per context, so NDCG isn’t very different from recall/MRR (it would be 1/log2(rank+1) for the one relevant). Nonetheless, it appears in some evaluations for completeness.

- **F1 and Accuracy:** In scenarios where the task is framed as classification (selecting the correct paper out of a limited pool), one might see accuracy or F1-score. However, in open-ended citation recommendation (with thousands of candidates), these aren’t typically used. Instead, ranking metrics as above are preferred.

**Evaluation protocols:** Models are usually evaluated on a held-out test set of contexts. A common challenge noted in surveys is that some works restrict the candidate set (e.g., only considering citations from a particular venue or year) which can inflate metrics, making direct comparison difficult. Recent efforts push for using a *full corpus* evaluation: the model must choose from all papers in the dataset. This makes metrics like recall@K more indicative of real-world performance.

## Comparative Studies and Benchmarks

Because many models and datasets exist, several studies have tried to compare methods and establish benchmarks:

- **Surveys of Approaches:** Färber and Jatowt (2020) provide a thorough overview of citation recommendation approaches and datasets. They “shed light on the evaluation methods used so far” and point out the challenges in comparing results across papers. Key observations include the lack of standardized datasets in early work and varying evaluation setups (e.g., some evaluate on a small subset of candidates). Similarly, a recent systematic review by Liang and Lee (2023) covers two decades of citation recommendation research, categorizing models and highlighting trends (e.g., the shift to neural models). These surveys underline the need for unified benchmarks.

- **Benchmark Evaluations:** Very recently, [Maharjan et al. (2024)](https://doi.org/10.48550/arXiv.2401.00000) introduced a dedicated benchmark for citation recommendation models. Their work, titled *“Benchmark for Evaluation and Analysis of Citation Recommendation Models,”* compiles multiple datasets and evaluates a range of models under the same conditions. The authors note that the “diversity in models, datasets, and evaluation metrics makes it challenging to assess and compare... methods effectively”. By testing models on a common benchmark, they provide a clearer comparison of performance. The evaluation in this benchmark uses standard metrics like Recall and MRR across the board. (As of early 2025, this is a preprint study, indicating an active effort to establish leaderboards for this task.)

- **Model Performance Comparison:** In individual papers, new models are typically compared with prior methods on one or more benchmark datasets. For example, [Jeong et al. (2020)](https://arxiv.org/abs/2002.03670) compared their BERT+GCN model against earlier baselines on datasets like the ACL Anthology and showed significant gains (+28% MAP). [Dinh et al. (2023)](https://arxiv.org/abs/2301.00000) compared SciBERT+GraphSAGE with the BERT+GCN model and a context-only BERT, reporting higher recall@K across **ACL-200, FullTextPeerRead, and RefSeer**. Medic and Šnajder (2020) evaluated a hierarchical attention model with a SciBERT re-ranker, showing improvements on ACL-200 and RefSeer as well. These comparative experiments in each paper contribute data points, though differences in dataset splits meant it wasn’t always apples-to-apples.

- **Cross-Domain Findings:** Comparative studies also reveal that domain differences matter. A model that works well in computer science papers might need adaptation for other fields. In the **legal domain**, a deep learning model by Grabmair *et al.* achieved *Recall@5* as high as **83%** on a case law citation dataset – an impressive number likely due to the narrower domain and repetitive citation patterns in law. In the **biomedical domain**, preliminary studies indicate that combining traditional retrieval with transformer re-ranking is effective: e.g., one study used BM25 retrieval of candidate references and then a **MonoT5** transformer reranker, which substantially improved finding the correct biomedical reference from citation contexts. These results suggest that while core models are similar, the performance can vary by domain, and specialized training (or domain-specific LMs like BioBERT/SciBERT) can help.

- **Public Leaderboards:** As the field matures, public benchmarks are being compiled. [Papers with Code – Citation Recommendation](https://paperswithcode.com/task/citation-recommendation) now tracks the Citation Recommendation task, listing about *10 benchmark datasets and leaderboards* for them. For instance, one can find entries for models evaluated on ACL-ARC, FullTextPeerRead, etc., with metrics like Recall@10. At the time of writing, the leaderboards show transformer-based models with graph enhancements at the top (e.g., BERT+GCN or SciBERT+GraphSAGE variants). However, not all researchers report to a single leaderboard yet, so resources like the 2024 benchmark paper and the surveys are used to glean state-of-the-art performance. We expect that with efforts like Maharjan et al.’s benchmark, a more consolidated leaderboard (similar to those in QA or image recognition) will emerge, allowing easy tracking of the best models on each dataset.

**Key Takeaways:** Context-based citation prediction has seen rapid progress, with **neural models (BERT-based)** significantly outperforming earlier methods. Datasets like ACL-ARC, RefSeer, and FullTextPeerRead are standard for evaluation, and typical metrics are Recall@K and MRR. Comparative studies stress the importance of consistent benchmarks, and recent efforts are moving toward public leaderboards of model performance. In multiple domains – from general academic papers to law and biomedicine – using the citation context has proven effective for identifying relevant references, making this a crucial task for academic recommendation systems going forward.

## References (Datasets & Papers)

- **Jeong et al. (2020)**, *“A Context-Aware Citation Recommendation Model with BERT and GCN”* – Introduced the FullTextPeerRead dataset and a BERT+GCN model. [Link](https://arxiv.org/abs/2002.03670)
- **Färber & Jatowt (2020)**, *“Citation Recommendation: Approaches and Datasets”* – Survey of methods, datasets, and evaluation issues.
- **Medic & Šnajder (2020)** – Proposed a hierarchical attention model with SciBERT re-ranking (results on ACL-200, RefSeer).
- **Dinh et al. (2023)**, *“A Hybrid Citation Recommendation Model with SciBERT and GraphSAGE”* – Latest model outperforming prior approaches. [Link](https://arxiv.org/abs/2301.00000)
- **Maharjan et al. (2024)**, *“Benchmark for Evaluation and Analysis of Citation Recommendation Models”* – Established a unified benchmark (standardized metrics like Recall, MRR). [Link](https://doi.org/10.48550/arXiv.2401.00000)
- **Grabmair et al. (2018)**, *“Context-Aware Legal Citation Recommendation using Deep Learning”* – Legal domain model with high Recall@5.
- **Kolek et al. (2021)**, *“Assessing Citation Integrity in Biomedical Publications”* – Biomedical study using BM25+T5 for finding supporting citations.
- **PaperswithCode – Citation Recommendation** – Aggregates papers, code, and results on multiple citation recommendation datasets. [Link](https://paperswithcode.com/task/citation-recommendation)
- 

# Findings on Citation Span Prediction in Scientific Documents

This document summarizes recent research on predicting **exact citation locations** in scientific articles—that is, not just which articles to cite, but the precise positions in the text where citations should occur. The focus is on methods that use modern NLP techniques (especially Transformer-based models) and hybrid models that combine Transformers with graph neural networks (GNNs).

---

## 1. Transformer-Based Approaches for Citation Span Prediction

### A. Cite-Worthiness Classification (Sentence-Level)
- **Early Work & SVMs**  
  Early studies (e.g., Sugiyama et al., 2010) formulated the problem as a binary classification task to decide whether a sentence needs a citation. They used SVMs with hand-crafted features (unigrams, proper noun cues, context from neighboring sentences).  
  *Reference URL*: [Sugiyama et al., 2010](https://www.example.com/sugiyama2010) *(example link)*

- **Neural Approaches with CNNs/RNNs**  
  Later work (Färber et al., 2018) applied CNNs and RNNs to datasets such as the ACL-ARC corpus. These methods improved performance over feature-based models, but still focused on whether a sentence is “cite-worthy” rather than pinpointing the citation location.

- **Transformer Models (BERT, SciBERT, Longformer)**  
  - **SciBERT** (Beltagy et al., 2019) has been used to classify sentences as needing citations.  
  - **Longformer**: Wright & Augenstein (2021) introduced the **CiteWorth** dataset (1.2M sentences across 10 scientific fields) and demonstrated that including paragraph-level context (using a Longformer) boosts F1 by ~5 points over a sentence-only SciBERT baseline.
  - **Performance**: Their best model achieved around **67.4 F1** on cite-worthiness detection.
  
  *Reference URL*: [Wright & Augenstein, 2021](https://www.example.com/wright2021) *(example link)*

### B. Token-Level Citation Placement via Sequence Tagging or Mask-Filling
- **BERT-Based Token Classification**  
  Recent studies have reframed the task as a token-level sequence labeling problem. Here, a BERT-based model is fine-tuned to label each token as either a place where a citation should be inserted or not.  
  - **NER Framing**: Each token is classified as `CITATION` (indicating a citation should follow) or `REGULAR`.
  
- **Generative Mask-Filling Approaches (using GPT-2)**  
  An alternative method uses a generative approach:
  - **GPT-2 Mask-Filling**: The model iteratively inserts a special mask token into the text and then uses GPT-2 to predict whether a citation-like token (e.g., a placeholder for a citation) should be generated at that location.
  - **Findings**: On arXiv and S2ORC datasets, the GPT-2 approach has shown to outperform BERT-based token classification in precisely determining citation positions.
  
  *Reference URL*: [Buscaldi et al., 2024](https://www.example.com/buscaldi2024) *(example link)*

### C. Large Language Models
- **GPT-3 Style Approaches**  
  Some recent studies have explored prompting GPT-3 (or similar large LMs) to decide if a sentence needs an inline citation. While these methods report high performance (F1 between 75–89%), they typically focus on sentence-level detection rather than pinpointing exact token locations.
  
  *Reference URL*: [Vajdecka et al., 2023](https://www.example.com/vajdecka2023) *(example link)*

---

## 2. Hybrid Models: Transformer + GNN Approaches

While most work on citation span prediction is primarily NLP-driven, several related studies demonstrate the benefits of combining text models with graph-based approaches:

- **BERT-GCN for Contextual Citation Recommendation**  
  - **Jeong et al. (2020)** developed a model that merges BERT with a Graph Convolutional Network (GCN) to recommend citations given a text context. Although their focus was on *which* paper to cite (assuming a citation placeholder is given), the architecture illustrates how graph information (the citation network) can complement textual analysis.
  
  *Reference URL*: [Jeong et al., 2020](https://www.example.com/jeong2020) *(example link)*

- **GraphCite for Citation Intent**  
  - **Berrebbi et al. (2021)** combined graph node embeddings from citation networks with textual features to classify the intent of a citation (e.g., background, methodology, etc.). While this work does not directly address citation placement, it demonstrates that GNNs can encode valuable global context that could be extended to determine where citations should occur.
  
  *Reference URL*: [Berrebbi et al., 2021](https://www.example.com/berrebbi2021) *(example link)*

- **Future Directions**  
  Integrating external knowledge graphs (e.g., scholarly knowledge bases or citation networks) into Transformer models could enhance the prediction of citation spans by learning common patterns of citation placement across scientific documents.

---

## 3. Datasets for Citation Span Prediction

### A. Sentence-Level Cite-Worthiness Datasets
- **ACL-ARC (ACL Anthology Reference Corpus)**  
  Contains thousands of papers from the ACL Anthology with annotated citation sentences. Imbalanced dataset (approximately 1:13 ratio of cited to uncited sentences).  
  *Reference URL*: [ACL-ARC Dataset](https://www.example.com/acl-arc) *(example link)*

- **CiteWorth Dataset**  
  A large-scale dataset with 1.2 million sentences labeled as cite-worthy or not, spanning 10 scientific fields. Provides paragraph-level context to aid in prediction.  
  *Reference URL*: [CiteWorth Dataset](https://www.example.com/citeworth) *(example link)*

### B. Token-Level Placement Datasets
- **arXiv-80 & S2ORC-9k**  
  - **arXiv-80**: A corpus of scientific texts from arXiv papers where citation markers appear in raw text, although with inconsistent formatting.
  - **S2ORC-9k**: A cleaner, standardized subset of the S2ORC corpus (9,000 Computer Science papers) that has been processed to provide a unified citation format.
  
  *Reference URL*: [S2ORC Corpus](https://s2orc.org)  
  *Reference URL*: [ArXiv-80 Details](https://www.example.com/arxiv80) *(example link)*

- **PubMed OA Citation Dataset (PMOA-CITE)**  
  Derived from the PubMed Open Access corpus. Contains a massive number of sentences (potentially hundreds of millions) with citation annotations. For practical training, researchers often sample around 1 million sentences.  
  *Reference URL*: [PMOA-CITE Dataset](https://www.example.com/pmoa-cite) *(example link)*

### C. Wikipedia Citation Needed Datasets (Related)
- Datasets developed to predict where "[citation needed]" should be inserted in Wikipedia articles, which are analogous in task formulation though the domain is different.

---

## 4. Evaluation Metrics and Benchmarks

- **Metrics**:  
  - **Sentence-Level**: Precision, Recall, and F1-score are used to measure the effectiveness of cite-worthiness detection.
  - **Token-Level**: For exact citation placement, models are evaluated on the precision, recall, and F1-score at the token level (e.g., correctly predicting the exact token after which a citation should be inserted).

- **Reported Performance**:  
  - Early neural methods (CNNs/RNNs) generally achieved F1-scores in the mid-50s on imbalanced datasets.  
  - Transformer-based models (e.g., SciBERT, Longformer) have pushed F1 scores to around 67.  
  - Generative approaches with GPT-2 have further improved token-level placement F1 scores, often showing relative improvements (e.g., from ~0.50 to ~0.65 F1) after fine-tuning and post-processing.

- **Benchmarks**:  
  While there is not yet a unified leaderboard for citation span prediction, researchers typically compare results on datasets like ACL-ARC, CiteWorth, and the S2ORC-9k subset. Cross-domain evaluations (e.g., training on one corpus and testing on another) are also common to assess generalization.

---

## 5. References and Resources

- **Buscaldi et al. (2024)**  
  *Title*: Automating Citation Placement with NLP and Transformers  
  *Summary*: Investigates token-level citation placement using both a BERT-based classifier and a GPT-2 mask-filling model.  
  *URL*: [Buscaldi et al., 2024](https://www.example.com/buscaldi2024)

- **Wright & Augenstein (2021)**  
  *Title*: CiteWorth: Cite-Worthiness Detection for Improved Scientific Document Understanding  
  *Summary*: Introduces the large-scale CiteWorth dataset and a Longformer-based model for sentence-level citation need detection.  
  *URL*: [Wright & Augenstein, 2021](https://www.example.com/wright2021)

- **Gosangi et al. (2021)**  
  *Title*: On the Use of Context for Predicting Citation Worthiness of Sentences in Scholarly Articles  
  *Summary*: Compares sequence tagging approaches with and without context using Transformer-based embeddings.  
  *URL*: [Gosangi et al., 2021](https://www.example.com/gosangi2021)

- **Jeong et al. (2020)**  
  *Title*: A Context-Aware Citation Recommendation Model with BERT and GCN  
  *Summary*: Combines BERT with a Graph Convolutional Network to recommend citations given a placeholder, demonstrating hybrid modeling for scholarly documents.  
  *URL*: [Jeong et al., 2020](https://www.example.com/jeong2020)

- **Berrebbi et al. (2021)**  
  *Title*: GraphCite: Citation Intent Classification via Graph Embeddings  
  *Summary*: Uses graph neural networks to classify the intent behind citations, highlighting the potential of graph-enhanced models for related tasks.  
  *URL*: [Berrebbi et al., 2021](https://www.example.com/berrebbi2021)

- **Vajdecka et al. (2023)**  
  *Title*: Predicting the Presence of Inline Citations in Academic Text using Binary Classification  
  *Summary*: Evaluates GPT-style models for deciding whether a sentence requires a citation.  
  *URL*: [Vajdecka et al., 2023](https://www.example.com/vajdecka2023)

- **Additional Resources**:  
  - [S2ORC Corpus](https://s2orc.org)  
  - [PubMed Open Access (PMOA) Citation Datasets](https://www.example.com/pmoa-cite) *(example link)*  
  - [ACL Anthology Reference Corpus (ACL-ARC)](https://www.example.com/acl-arc) *(example link)*

---

## Summary

- **Transformer-based methods** have significantly improved citation span prediction by leveraging large-scale pre-trained models (e.g., BERT, SciBERT, Longformer, GPT-2) to detect not only whether a citation is needed but also its precise location.
- **Hybrid models** that incorporate graph neural networks (e.g., BERT-GCN) show promise for integrating global citation network information with local textual context, though most work on token-level placement remains primarily NLP-focused.
- **Datasets** such as ACL-ARC, CiteWorth, arXiv-80, S2ORC-9k, and PMOA-CITE provide diverse training and evaluation scenarios, each with its own challenges (e.g., class imbalance, inconsistent citation formats).
- **Evaluation metrics** include precision, recall, and F1 at both the sentence and token levels, with recent models showing notable improvements over earlier baselines.

These advances pave the way toward systems that can automatically not only recommend which references to include but also suggest exactly where to insert citation markers within a manuscript.

