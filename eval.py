import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import logging
import pickle

from wiki_processor import WikiDumpProcessor
from config import TrainingConfig

@dataclass
class SearchResult:
    """Represents a single search result with title and similarity score."""
    title: str
    score: float
    text_preview: str

class WikiEmbeddingSearch:
    """Handles embedding computation and similarity search for Wikipedia articles."""
    
    def __init__(self, 
                 config: TrainingConfig,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 batch_size: int = 64):
        self.config = config
        self.model = SentenceTransformer(model_name)
        self.model.to(self.config.device)
        self.max_length = self.model.max_seq_length
        self.batch_size = batch_size
        
        # Initialize storage for embeddings and article mapping
        self.article_embeddings = None
        self.article_index = None  # FAISS index
        self.title_to_idx = {}
        self.idx_to_title = {}
        self.idx_to_preview = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_wiki_dump(self, dump_path: str, cache_dir: Optional[str] = None):
        """Process Wikipedia dump and compute embeddings for all articles with batch processing."""
        self.logger.info(f"Processing Wikipedia dump from {dump_path}")
        
        # Initialize processor
        processor = WikiDumpProcessor(dump_path)
        
        # First count total articles
        total_articles = sum(1 for _ in processor.iter_articles(skip_redirects=True))
        self.logger.info(f"Found {total_articles} articles to process")
        
        # Initialize batch storage
        current_batch = []
        current_batch_titles = []
        current_batch_previews = []
        idx = 0
        
        # Process articles in batches
        for article in tqdm(processor.iter_articles(skip_redirects=True), 
                          total=total_articles,
                          desc="Processing articles"):
            # Clean and truncate text
            clean_text = article.text[:self.max_length]
            if not clean_text.strip():
                continue
            
            # Add to current batch
            current_batch.append(clean_text)
            current_batch_titles.append(article.title)
            current_batch_previews.append(clean_text[:200] + "...")
            
            # Process batch when it reaches batch_size
            if len(current_batch) >= self.batch_size:
                self._process_batch(current_batch, current_batch_titles, current_batch_previews, idx)
                idx += len(current_batch)
                current_batch = []
                current_batch_titles = []
                current_batch_previews = []
        
        # Process any remaining articles in the last batch
        if current_batch:
            self._process_batch(current_batch, current_batch_titles, current_batch_previews, idx)
            idx += len(current_batch)
        
        # Build FAISS index
        self._build_faiss_index()
        
        # Save to cache if specified
        if cache_dir:
            self.save_cache(cache_dir)
            
        self.logger.info(f"Processed {len(self.title_to_idx)} articles")

    def _process_batch(self, texts: List[str], titles: List[str], previews: List[str], start_idx: int):
        """Process a batch of articles and compute their embeddings."""
        # Compute embeddings for the batch
        with torch.no_grad():
            batch_embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=self.batch_size
            )
        
        # Convert to numpy and store
        batch_embeddings_np = batch_embeddings.cpu().numpy()
        
        # Initialize or append to article_embeddings
        if self.article_embeddings is None:
            self.article_embeddings = batch_embeddings_np
        else:
            self.article_embeddings = np.vstack([self.article_embeddings, batch_embeddings_np])
        
        # Store mappings
        for i, (title, preview) in enumerate(zip(titles, previews)):
            idx = start_idx + i
            self.title_to_idx[title] = idx
            self.idx_to_title[idx] = title
            self.idx_to_preview[idx] = preview

    def _build_faiss_index(self):
        """Build FAISS index for efficient similarity search."""
        dimension = self.article_embeddings.shape[1]
        self.article_index = faiss.IndexFlatIP(dimension)  # Inner product index
        self.article_index.add(self.article_embeddings.astype(np.float32))

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search for most similar articles to the query."""
        # Compute query embedding
        with torch.no_grad():
            query_embedding = self.model.encode(
                query,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        
        # Perform similarity search
        scores, indices = self.article_index.search(
            query_embedding.cpu().numpy().reshape(1, -1).astype(np.float32),
            k
        )
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            title = self.idx_to_title[idx]
            preview = self.idx_to_preview[idx]
            results.append(SearchResult(title=title, score=float(score), text_preview=preview))
        
        return results

    def save_cache(self, cache_dir: str):
        """Save computed embeddings and mappings to cache."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings and mappings
        np.save(cache_path / "embeddings.npy", self.article_embeddings)
        with open(cache_path / "mappings.pkl", "wb") as f:
            pickle.dump({
                "title_to_idx": self.title_to_idx,
                "idx_to_title": self.idx_to_title,
                "idx_to_preview": self.idx_to_preview
            }, f)
        
        self.logger.info(f"Cached embeddings and mappings to {cache_dir}")

    def load_cache(self, cache_dir: str):
        """Load computed embeddings and mappings from cache."""
        cache_path = Path(cache_dir)
        
        # Load embeddings
        self.article_embeddings = np.load(cache_path / "embeddings.npy")
        
        # Load mappings
        with open(cache_path / "mappings.pkl", "rb") as f:
            mappings = pickle.load(f)  # Fixed typo: pickle.dump -> pickle.load
            self.title_to_idx = mappings["title_to_idx"]
            self.idx_to_title = mappings["idx_to_title"]
            self.idx_to_preview = mappings["idx_to_preview"]
        
        # Rebuild FAISS index
        self._build_faiss_index()
        
        self.logger.info(f"Loaded cache from {cache_dir}")

def main():
    # Initialize config
    config = TrainingConfig()
    config.device = 'cuda:1'
    
    # Set up paths
    wiki_dump_path = config.raw_data_dir / "wiki" / "simplewiki-latest-pages-articles.xml.bz2"
    cache_dir = config.cache_dir / "embeddings"
    
    # Initialize search system
    search = WikiEmbeddingSearch(config)
    
    # Check if cache exists
    if (cache_dir / "embeddings.npy").exists():
        print("Loading embeddings from cache...")
        search.load_cache(cache_dir)
    else:
        print("Processing Wikipedia dump and computing embeddings...")
        search.process_wiki_dump(str(wiki_dump_path), str(cache_dir))
    
    # Example search
    query = "What is the theory of relativity?"
    results = search.search(query, k=5)
    
    print(f"\nTop 5 results for query: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.title} (score: {result.score:.3f})")
        print(f"   Preview: {result.text_preview}\n")

if __name__ == "__main__":
    main()