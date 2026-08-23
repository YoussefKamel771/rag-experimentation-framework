from src.pipeline.indexer_pipeline import BEIRIndexer
from configs.config import get_settings

config = get_settings("configs/config.yaml")

indexer = BEIRIndexer(config, 
                      dataset="scifact",
                      beir_data_dir="data/scifact")
manifest = indexer.build()

# keep the loader around for eval — it holds queries + qrels
# queries = indexer.loader.get_queries()
# qrels = indexer.loader.get_qrels()