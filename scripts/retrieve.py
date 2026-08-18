import argparse

from configs.config import get_settings, Settings
from src.pipeline.retrieval_pipeline import retrieve

def run_retrieval(query: str, config: Settings):
    
    results = retrieve(config, query)

    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}\n")

    for result in results:
        print("=" * 80)
        print(
            f"Rank: {result.rank} "
            f"Score: {result.score:.6f} "
            f"Via: {result.retriever} "
            f"ID: {result.chunk.chunk_id}"
        )
        print(result.metadata)
        print("-" * 80)
        print(result.chunk.text[:1500])
        print()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default="configs/config.yaml")
    # parser.add_argument("--query", required=True)
    # args = parser.parse_args()

    config = get_settings(config_path="configs/config.yaml")
    run_retrieval(query="what are ducks?", config=config)
