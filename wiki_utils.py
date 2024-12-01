# Standard library imports
import bz2
import json
import logging
import osimport re
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

# Third-party imports

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



class CitationExtractor:
    """Prepares citation data for model training."""

    def __init__(self, jsonl_path: str = "data/wiki_articles.jsonl"):
        
        # Load articles
        logging.info("Loading articles from JSONL file...")
        self.articles_dict = {}
        self.id2ref = {}
        self.ref2id = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                article = json.loads(line)
                ref = article['title'].lower()
                id = len(self.articles_dict) + 1
                self.articles_dict[ref] = self.sanitize_wiki_content(article['text'])
                self.ref2id[ref] = id 
                self.id2ref[id] = ref
        logging.info(f"Loaded {len(self.articles_dict)} articles.")

    def extract_citation_spans(self,text):
        citations = []
        for match in re.finditer(r'\[\[(.*?)\]\]', text):
            match_text = match.group(1)
            citation = match_text.split('|') if '|' in match_text else [match_text]
            citation = [(c.split('#')[0] if '#' in c else c) for c in citation]
            ref = None
            for cit in citation:
                if cit.lower() in self.articles_dict:
                    ref = cit.lower()
                    break
            if ref:
                citations.append((match.start(), match.end(), self.ref2id[ref]))
        return citations

    @staticmethod
    def sanitize_wiki_content(text: str) -> str:
        """Cleans wiki content by removing metadata and formatting."""
        # Find main content starting from first bold title
        match = re.search(r"'''([^']+?)'''", text)
        if match:
            text = text[match.start():]

        # Remove wiki elements and clean up
        text = re.sub(r'\[\[File:.*\]\]|\[\[Category:.*\]\]|\{\{stub.*\}\}', '', text)
        return '\n'.join(line for line in text.split('\n') if line.strip())

    def find_source_citations(self) -> Tuple[List[str], List[Tuple[List[str], int, int]]]:
        """Creates source-target pairs for citation matching."""

        articles = list(self.articles_dict.keys())
        sources = []
        citation_data = []

        for title in articles:
            text = self.articles_dict[title]
            source_text = self.sanitize_wiki_content(text)
            citations = self.extract_citation_spans(source_text)            
            sources.append(source_text)
            citation_data.append(citations)

        return sources, citation_data