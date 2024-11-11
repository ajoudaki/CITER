# Citation Matcher

This repository contains a deep learning model for matching citations in Wikipedia articles. The system learns to identify and link appropriate citations by understanding the context in which they appear and the content of the cited articles. The main idea is to implement this as a retrieval: we want the model to be able to embed the citing page (with the text of citation masked) and the target in a vector space, such that these vectors are close. Thus, given a page with a reference masked, we can retrieve pages that are likely to be at that location. 

## Predicting citations as a retrieval problem
- Data processing
  - Processes Wikipedia XML dumps efficiently
  - Extracts and cleans article content and citations
  - For each reference, creating a pair of source (citing page) and target (cited page) pages
  - After pre-processing, it can store it in JSONL and SQLite storage formats
- Language modeling
  - Define a special token for citation `<CITE>` which replaces and masks the reference in source, and for a refernece `<REF>`, which is added to the ened of target page
  - Implements a language model based citation matching, that learns to make embedding of matching <CITE> and <REF> tokens as close as possible.
- Evaluation:
  - We treat this as a retrieval problem: Given a validation set of $n$ source and target pairs, we compute $n$ source and $N$ target vectors, normalize them, and compute their pairwise inner products, and then compute the cross entropy loss, assuming that for the row $i$, the correct label is $i$.
  - Metrics:
    - Top-k accuracy: the recall of finding the right target, if we look at the first $k$ closest vectors
    - Mean Reciprocal Rank: the mean value of $\frac1N \sum_i 1/r_i$ where, where $r_i$ is the rank of the true target for source $i$

```mermaid
graph TD
    subgraph "Source Pages"
        S1["The cat is a domestic<br/>species of [&lt;CITE&gt;] mammal"]
        S2["Mount Everest is in<br/>the [&lt;CITE&gt;] range"]
    end
    subgraph "Similarity Matrix"
        M["     v₁     v₂    <br/>u₁  0.89   0.15<br/>u₂  0.12   0.92"]
    end
    subgraph "Target Pages"
        T1["Mammals are vertebrate<br/>animals with hair. [&lt;REF&gt;]"]
        T2["The Himalayas contain the<br/>world's highest peaks. [&lt;REF&gt;]"]
    end
    S1 --> M
    S2 --> M
    M --> T1
    M --> T2
    classDef sourceClass fill:#e3f2fd,stroke:#1565c0
    classDef targetClass fill:#e8f5e9,stroke:#2e7d32
    classDef matrixClass fill:#fff3e0,stroke:#f57c0
    class S1,S2 sourceClass
    class T1,T2 targetClass
    class M matrixClass
```
