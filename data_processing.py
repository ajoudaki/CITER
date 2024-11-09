# Data Processing Module
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
    
    NAMESPACE = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}
    EXCLUDED_PREFIXES = {
        'Wikipedia:', 'Template:', 'Category:', 'Portal:',
        'File:', 'MediaWiki:', 'Help:', 'Book:', 'Draft:',
        'TimedText:', 'Module:', 'Special:'
    }

    def __init__(self, dump_path: str):
        self.dump_path = dump_path

    def _open_dump(self) -> Iterator:
        """Opens the dump file, handling bz2 compression if needed."""
        if self.dump_path.endswith('.bz2'):
            return bz2.BZ2File(self.dump_path)
        return open(self.dump_path, 'rb')

    def _extract_page_data(self, page_elem: ET.Element) -> WikiArticle:
        """Extracts article data from a page XML element."""
        title_elem = page_elem.find('.//mw:title', self.NAMESPACE)
        revision = page_elem.find('.//mw:revision', self.NAMESPACE)
        
        text = ''
        timestamp = ''
        if revision is not None:
            text_elem = revision.find('mw:text', self.NAMESPACE)
            timestamp_elem = revision.find('mw:timestamp', self.NAMESPACE)
            text = text_elem.text if text_elem is not None else ''
            timestamp = timestamp_elem.text if timestamp_elem is not None else ''
        
        title = title_elem.text if title_elem is not None else ''
        is_redirect = bool(re.match(r'#REDIRECT', text or '', re.IGNORECASE))
        
        return WikiArticle(title=title, text=text, timestamp=timestamp, is_redirect=is_redirect)

    def iter_articles(self, skip_redirects: bool = True) -> Iterator[WikiArticle]:
        """Iterates through valid articles in the dump."""
        context = ET.iterparse(self._open_dump(), events=('end',))
        
        for _, elem in context:
            if not elem.tag.endswith('page'):
                continue
                
            article = self._extract_page_data(elem)
            
            if any(article.title.startswith(prefix) for prefix in self.EXCLUDED_PREFIXES):
                elem.clear()
                continue
                
            if skip_redirects and article.is_redirect:
                elem.clear()
                continue
                
            yield article
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
                    (title TEXT PRIMARY KEY,
                     text TEXT,
                     timestamp TEXT,
                     is_redirect INTEGER)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_title ON articles(title)')
        
        count = 0
        batch = []
        
        try:
            for i, article in enumerate(self.processor.iter_articles()):
                if sample_size is not None and i >= sample_size:
                    break
                    
                batch.append((
                    article.title,
                    article.text,
                    article.timestamp,
                    1 if article.is_redirect else 0
                ))
                
                if len(batch) >= batch_size:
                    self._execute_batch_insert(c, batch)
                    conn.commit()
                    count += len(batch)
                    batch = []
            
            if batch:
                self._execute_batch_insert(c, batch)
                conn.commit()
                count += len(batch)
                
        finally:
            conn.close()
            
        return count

    @staticmethod
    def _execute_batch_insert(cursor: sqlite3.Cursor, batch: List[Tuple]):
        """Executes a batch insert into the SQLite database."""
        cursor.executemany(
            'INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?)',
            batch
        )

class ArticleContentProcessor:
    """Processes Wikipedia article content for citation matching."""
    
    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """Extracts and cleans citations from article text."""
        citations = re.findall(r'\[\[(.*?)\]\]', text)
        return [citation.split('|')[0] for citation in citations]

    @staticmethod
    def clean_wiki_content(text: str) -> str:
        """Cleans wiki content by removing metadata and formatting."""
        # Find the main article content
        title_match = re.search(r"'''([^']+?)'''", text)
        if not title_match:
            return text
            
        start_pos = title_match.start()
        content = text[start_pos:]
        
        # Remove various wiki elements
        patterns = [
            (r'\[\[Category:.*?\]\]', ''),  # Categories
            (r'\[\[File:.*?\]\]', ''),      # Files
            (r'\{\{stub\}\}', '')           # Stub templates
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # Clean up empty lines
        content = '\n'.join(line for line in content.split('\n') if line.strip())
        return content.strip()

class CitationDataPreprocessor:
    """Prepares citation data for model training."""
    
    def __init__(self, articles_dict: Dict[str, str]):
        self.articles_dict = articles_dict

    def create_citation_pairs(self, sample_size: int = 1000, cite_samples_per_article: int = 1) -> Tuple[List[str], List[str]]:
        """Creates source-target pairs for citation matching."""
        articles = np.random.permutation(list(self.articles_dict.keys()))[:sample_size]
        sources = []
        targets = []
        
        for article_title in articles:
            source_text = self.articles_dict[article_title]
            citations = ArticleContentProcessor.extract_citations(source_text)
            valid_citations = [c for c in citations if c.lower() in self.articles_dict]
            
            if not valid_citations:
                continue
                
            sample_size = min(cite_samples_per_article, len(valid_citations))
            sampled_citations = np.random.choice(valid_citations, sample_size, replace=False)
            
            for citation in sampled_citations:
                try:
                    source_content = self._prepare_source_context(source_text, citation)
                    target_content = self._prepare_target_content(citation)
                    sources.append(source_content)
                    targets.append(target_content)
                except Exception:
                    continue
        
        return sources, targets

    def _prepare_source_context(self, text: str, citation: str) -> str:
        """Prepares source context with citation markers."""
        text = ArticleContentProcessor.clean_wiki_content(text)
        return text.replace(f"[[{citation}]]", "<CITE>")

    def _prepare_target_content(self, citation: str) -> str:
        """Prepares target content with reference marker."""
        content = self.articles_dict[citation.lower()]
        return f"{ArticleContentProcessor.clean_wiki_content(content)} <REF>"