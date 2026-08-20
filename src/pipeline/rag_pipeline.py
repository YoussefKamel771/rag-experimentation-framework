from __future__ import annotations

from configs.config import Settings

from src.plugins.registry_loader import load_all_plugins
from src.pipeline.retrieval_pipeline import create_retriever
from src.pipeline.reranking_pipeline import create_reranker
from src.pipeline.generation_pipeline import create_context_builder, create_generator, build_sources


class RAGPipeline:
    def __init__(self, config: Settings):
        load_all_plugins()
        self.config = config
        self.retriever = create_retriever(config)
        self.reranker = create_reranker(config)
        self.context_builder = create_context_builder(config)
        self.generator = create_generator(config)

    def run(self, query: str) -> dict:
        candidates = self.retriever.retrieve(query, top_k=self.config.retrieval.candidate_k)
        reranked = self.reranker.rerank(query, candidates, top_k=self.config.reranker.top_k)
        context = self.context_builder.build(reranked)
        sources = build_sources(context)
        generation = self.generator.generate(query=query, context=context.text, sources=sources)
        return {"query": query,
                "retrieval_results": reranked,
                "context": context,
                "generation": generation}