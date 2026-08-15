from src.ingestion.models import Chunk, Document
from src.plugins.registry import chunker_registry
from .base import Chunker
from .utils import make_chunk

@chunker_registry.register("fixed")
class FixedChunker(Chunker):
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        if chunk_size<=0 or not 0<=chunk_overlap<chunk_size: raise ValueError("Invalid chunk_size/chunk_overlap")
        self.chunk_size=chunk_size; self.chunk_overlap=chunk_overlap
    def chunk(self, document: Document):
        out=[]; start=0; i=0; step=self.chunk_size-self.chunk_overlap
        while start<len(document.text):
            out.append(make_chunk(document,document.text[start:start+self.chunk_size],i,"fixed",{"chunk_size":self.chunk_size,"chunk_overlap":self.chunk_overlap}))
            i+=1; start+=step
        return out
