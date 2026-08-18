from src.pipeline.indexer import build_index
from configs.config import get_settings

def run_indexing() -> None:
    config = get_settings()
    build_index(config)

if __name__ == "__main__":
    run_indexing()
