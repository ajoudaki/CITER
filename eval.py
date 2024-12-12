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
import json
import re

from wiki_processor import WikiDumpProcessor
from config import TrainingConfig

@dataclass
class SearchResult:
    """Represents a single search result with title and similarity score."""
    title: str
    score: float
    text_preview: str


class WikiEmbeddingSearch:
    def __init__(self, 
                 config: TrainingConfig,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 batch_size: int = 64):
        self.config = config
        self.model_name = model_name
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
        
        # Add storage for reference text and links
        self.reference_sanitized = {}
        self.reference_links = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def save_cache(self, cache_dir: str):
        cache_path = self._get_model_cache_dir(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Save model information and embeddings
        cache_info = {
            "model_name": self.model_name,
            "embedding_dim": self.article_embeddings.shape[1],
            "num_articles": len(self.title_to_idx)
        }
        
        with open(cache_path / "cache_info.json", "w") as f:
            json.dump(cache_info, f)
            
        np.save(cache_path / "embeddings.npy", self.article_embeddings)
        
        # Save all mappings including reference data
        with open(cache_path / "mappings.pkl", "wb") as f:
            pickle.dump({
                "title_to_idx": self.title_to_idx,
                "idx_to_title": self.idx_to_title,
                "idx_to_preview": self.idx_to_preview,
                "reference_sanitized": self.reference_sanitized,
                "reference_links": self.reference_links
            }, f)
        
        self.logger.info(f"Cached embeddings and reference data for model {self.model_name} to {cache_path}")

    def load_cache(self, cache_dir: str):
        cache_path = self._get_model_cache_dir(cache_dir)
        
        if not cache_path.exists():
            raise FileNotFoundError(f"No cache found for model {self.model_name} in {cache_path}")
        
        # Load model information
        with open(cache_path / "cache_info.json", "r") as f:
            cache_info = json.load(f)
        
        self.article_embeddings = np.load(cache_path / "embeddings.npy")
        
        # Load all mappings including reference data
        with open(cache_path / "mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
            self.title_to_idx = mappings["title_to_idx"]
            self.idx_to_title = mappings["idx_to_title"]
            self.idx_to_preview = mappings["idx_to_preview"]
            self.reference_sanitized = mappings.get("reference_sanitized", {})
            self.reference_links = mappings.get("reference_links", {})
        
        self._build_faiss_index()
        
        self.logger.info(f"Loaded cache for model {self.model_name} from {cache_path}")

    def process_wiki_dump(self, dump_path: str, cache_dir: Optional[str] = None):
        """Process Wikipedia dump with two-stage sanitization."""
        self.logger.info(f"Processing Wikipedia dump from {dump_path}")
        
        # Define cache paths
        if cache_dir:
            cache_dir = Path(cache_dir)
            basic_cache_path = cache_dir / "basic_sanitized.pkl"
            reference_cache_path = cache_dir / "reference_sanitized.pkl"
            links_cache_path = cache_dir / "reference_links.pkl"
        else:
            basic_cache_path = reference_cache_path = links_cache_path = None
    
        # Stage 1: Basic sanitization
        if basic_cache_path and basic_cache_path.exists():
            self.logger.info("Loading basic sanitized pages from cache...")
            with open(basic_cache_path, 'rb') as f:
                basic_sanitized = pickle.load(f)
        else:
            basic_sanitized = {}
            processor = WikiDumpProcessor(dump_path)
            for article in tqdm(processor.iter_articles(skip_redirects=True), desc="Basic sanitization"):
                clean_text = processor.sanitize_wiki_content(article.text)
                if clean_text.strip():
                    basic_sanitized[article.title] = clean_text
            
            if basic_cache_path:
                basic_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(basic_cache_path, 'wb') as f:
                    pickle.dump(basic_sanitized, f)
    
        # Stage 2: Reference sanitization with link tracking
        if reference_cache_path and reference_cache_path.exists() and links_cache_path.exists():
            self.logger.info("Loading reference sanitized pages from cache...")
            with open(reference_cache_path, 'rb') as f:
                self.reference_sanitized = pickle.load(f)
            with open(links_cache_path, 'rb') as f:
                self.reference_links = pickle.load(f)
        else:
            self.reference_sanitized = {}
            self.reference_links = {}
            for title, text in tqdm(basic_sanitized.items(), desc="Reference sanitization"):
                clean_text, links = self._sanitize_references(text)
                self.reference_sanitized[title] = clean_text
                self.reference_links[title] = links
            
            if reference_cache_path:
                with open(reference_cache_path, 'wb') as f:
                    pickle.dump(self.reference_sanitized, f)
                with open(links_cache_path, 'wb') as f:
                    pickle.dump(self.reference_links, f)
    
        # Process for embeddings
        current_batch = []
        current_batch_titles = []
        current_batch_previews = []
        idx = 0
        
        for title, text in tqdm(self.reference_sanitized.items(), desc="Processing embeddings"):
            truncated_text = text[:self.max_length]
            current_batch.append(truncated_text)
            current_batch_titles.append(title)
            current_batch_previews.append(truncated_text[:200] + "...")
            
            if len(current_batch) >= self.batch_size:
                self._process_batch(current_batch, current_batch_titles, current_batch_previews, idx)
                idx += len(current_batch)
                current_batch = []
                current_batch_titles = []
                current_batch_previews = []
        
        if current_batch:
            self._process_batch(current_batch, current_batch_titles, current_batch_previews, idx)
        
        self._build_faiss_index()
        
        if cache_dir:
            self.save_cache(str(cache_dir))

    def _get_model_cache_dir(self, cache_dir: str) -> Path:
        """Get model-specific cache directory."""
        # Create a safe directory name from model name
        model_dir = self.model_name.replace('/', '_')
        cache_path = Path(cache_dir) / model_dir
        return cache_path


    def _sanitize_references(self, text: str) -> tuple[str, list[tuple[int, int, str]]]:
        """Replace wiki references with their display text and collect link positions.
        
        Returns:
            tuple: (sanitized_text, [(start, end, link), ...])
            where start/end are positions in sanitized text
        """
        result = ""
        last_end = 0
        link_positions = []
        
        for match in re.finditer(r'\[\[(.*?)\]\]', text):
            # Add text before the reference
            result += text[last_end:match.start()]
            current_pos = len(result)  # This will be our start position
            
            # Process the reference
            match_text = match.group(1)
            if '|' in match_text:
                link, displayed_text = match_text.split('|', 1)
            else:
                link = displayed_text = match_text.split('#')[0]
                
            # Add the displayed text and record position
            result += displayed_text
            link_positions.append((current_pos, current_pos + len(displayed_text), link))
            
            last_end = match.end()
        
        # Add remaining text
        result += text[last_end:]
        return result, link_positions
            

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


import random
from typing import Dict, List, Tuple

class WikiMaskingExperiment:
    def __init__(self, search_instance):
        self.search = search_instance
        
    def mask_and_test(self, article_title: str) -> Tuple[str, str, List[SearchResult]]:
        """
        Masks a random link in the given article and tests retrieval.
        
        Returns:
            Tuple of (masked_text, masked_link, search_results)
        """
        # Get article text and links
        article_text = self.search.reference_sanitized[article_title]
        article_links = self.search.reference_links[article_title]
        
        if not article_links:
            raise ValueError("No links found in article")
            
        # Get unique links
        unique_links = list(set(link for _, _, link in article_links))
        
        # Randomly select a link to mask
        link_to_mask = random.choice(unique_links)
        
        # Create masked version
        masked_text = article_text
        positions = sorted(
            [(start, end) for start, end, link in article_links if link == link_to_mask],
            reverse=True  # Process from end to start to preserve positions
        )
        
        for start, end in positions:
            masked_text = masked_text[:start] + "[[MASKED]]" + masked_text[end:]
            
        # Perform search with masked text
        results = self.search.search(masked_text, k=10)
        
        return masked_text, link_to_mask, results

def run_masking_experiment(search_instance):
    experiment = WikiMaskingExperiment(search_instance)

    print(len(search_instance.reference_sanitized))
    
    # Randomly select an article
    article_title = random.choice(list(search_instance.reference_sanitized.keys()))
    
    try:
        masked_text, masked_link, results = experiment.mask_and_test(article_title)
        
        print(f"Selected article: {article_title}")
        print(f"\nMasked link: {masked_link}")
        print(f"\nMasked text preview: {masked_text[:500]}...")
        print("\nSearch results:")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.title} (score: {result.score:.3f})")
            print(f"   Preview: {result.text_preview}")
            
    except ValueError as e:
        print(f"Error processing article {article_title}: {e}")
        

def main():
    # Initialize config
    config = TrainingConfig()
    
    # Set up paths
    wiki_dump_path = config.raw_data_dir / "wiki" / "simplewiki-latest-pages-articles.xml.bz2"
    cache_dir = config.cache_dir / "embeddings"
    
    # Initialize search system
    search = WikiEmbeddingSearch(config)
    
    # Check if cache exists for this model
    model_cache_dir = search._get_model_cache_dir(cache_dir)
    if model_cache_dir.exists():
        print(f"Loading embeddings for model {search.model_name} from cache...")
        search.load_cache(cache_dir)
    else:
        print("Processing Wikipedia dump and computing embeddings...")
        search.process_wiki_dump(str(wiki_dump_path), str(cache_dir))

    run_masking_experiment(search)
    
    # # Example search
    # query = "Which animal is a reptile without legs that is dangerous? "
    # results = search.search(query, k=10)
    
    # print(f"\nTop 5 results for query: {query}\n")
    # for i, result in enumerate(results, 1):
    #     print(f"{i}. {result.title} (score: {result.score:.3f})")
        # print(f"   Preview: {result.text_preview}\n")

if __name__ == "__main__":
    main()