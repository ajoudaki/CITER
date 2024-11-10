from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterator, Union, Optional
import xml.etree.ElementTree as ET
import bz2
import json
import sqlite3
from pathlib import Path
import re
import numpy as np

@dataclass
class WikiArticle:
    """Represents a Wikipedia article with its metadata."""
    title: str
    text: str
    timestamp: str
    is_redirect: bool

class WikiDumpProcessor:
    """Processes Wikipedia XML dumps and extracts articles."""
    
    def __init__(self, dump_path: str):
        self.dump_path = dump_path
        self._ns = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}
        self._skip_prefixes = {
            'Wikipedia:', 'Template:', 'Category:', 'Portal:', 'File:', 
            'MediaWiki:', 'Help:', 'Book:', 'Draft:', 'TimedText:', 
            'Module:', 'Special:'
        }

    def iter_articles(self, skip_redirects: bool = True) -> Iterator[WikiArticle]:
        """Iterates through valid articles in the dump."""
        dump_file = bz2.BZ2File(self.dump_path) if self.dump_path.endswith('.bz2') else open(self.dump_path, 'rb')
        
        for _, elem in ET.iterparse(dump_file, events=('end',)):
            if not elem.tag.endswith('page'):
                continue

            # Extract basic article data
            title = elem.find('.//mw:title', self._ns).text
            if any(title.startswith(prefix) for prefix in self._skip_prefixes):
                elem.clear()
                continue

            # Get revision data
            rev = elem.find('.//mw:revision', self._ns)
            text = rev.find('mw:text', self._ns).text if rev is not None else ''
            timestamp = rev.find('mw:timestamp', self._ns).text if rev is not None else ''
            is_redirect = bool(re.match(r'#REDIRECT', text or '', re.IGNORECASE))

            if skip_redirects and is_redirect:
                elem.clear()
                continue

            yield WikiArticle(title=title, text=text, timestamp=timestamp, is_redirect=is_redirect)
            elem.clear()

class ArticleStorage:
    """Handles storage and retrieval of Wikipedia articles."""
    
    def __init__(self, processor: WikiDumpProcessor):
        self.processor = processor

    def save_to_jsonl(self, output_path: Union[str, Path], sample_size: Optional[int] = None) -> int:
        """Saves articles to a JSONL file."""
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, article in enumerate(self.processor.iter_articles()):
                if sample_size is not None and i >= sample_size:
                    break
                json.dump(article.__dict__, f, ensure_ascii=False)
                f.write('\n')
                count += 1
        return count

    def save_to_sqlite(self, db_path: Union[str, Path], sample_size: Optional[int] = None,
                      batch_size: int = 1000) -> int:
        """Saves articles to a SQLite database."""
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS articles
                    (title TEXT PRIMARY KEY, text TEXT, timestamp TEXT, is_redirect INTEGER)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_title ON articles(title)')
        
        count = 0
        batch = []
        
        try:
            for i, article in enumerate(self.processor.iter_articles()):
                if sample_size is not None and i >= sample_size:
                    break
                    
                batch.append((article.title, article.text, article.timestamp, 
                            1 if article.is_redirect else 0))
                
                if len(batch) >= batch_size:
                    c.executemany('INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?)', batch)
                    conn.commit()
                    count += len(batch)
                    batch = []
            
            if batch:
                c.executemany('INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?)', batch)
                conn.commit()
                count += len(batch)
                
        finally:
            conn.close()
            
        return count

class CitationDataPreprocessor:
    """Prepares citation data for model training."""
    
    def __init__(self, articles_dict: Dict[str, str]):
        self.articles_dict = articles_dict

    def create_citation_pairs(self, sample_size: int = 1000, cite_samples_per_article: int = 1) -> Tuple[List[str], List[str]]:
        """Creates source-target pairs for citation matching."""
        articles = np.random.permutation(list(self.articles_dict.keys()))[:sample_size]
        sources, targets = [], []
        
        for title in articles:
            text = self.articles_dict[title]
            citations = [cit.split('|')[0] for cit in re.findall(r'\[\[(.*?)\]\]', text)]
            valid_citations = [c for c in citations if c.lower() in self.articles_dict]
            
            if not valid_citations:
                continue
                
            for citation in np.random.choice(valid_citations, 
                                           min(cite_samples_per_article, len(valid_citations)), 
                                           replace=False):
                try:
                    # Clean and prepare content
                    source_text = self._clean_wiki_text(text)
                    source_text = source_text.replace(f"[[{citation}]]", "<CITE>")
                    target_text = f"{self._clean_wiki_text(self.articles_dict[citation.lower()])} <REF>"
                    
                    sources.append(source_text)
                    targets.append(target_text)
                except Exception:
                    continue
        
        return sources, targets

    @staticmethod
    def _clean_wiki_text(text: str) -> str:
        """Cleans wiki content by removing metadata and formatting."""
        # Find main content starting from first bold title
        match = re.search(r"'''([^']+?)'''", text)
        if match:
            text = text[match.start():]
        
        # Remove wiki elements and clean up
        text = re.sub(r'\[\[Category:.*?\]\]|\[\[File:.*?\]\]|\{\{stub\}\}', '', text)
        return '\n'.join(line for line in text.split('\n') if line.strip())