from config import (
    JD_PATH, CANDIDATES_PATH, JD_ANALYSIS_PATH, AGENT_AUDIT_PATH,
    RANKED_JSON_PATH, FINAL_CSV_PATH, USE_LOCAL_EMBEDDINGS, USE_DEV_AGENTS,
    TOP_K_FOR_AGENT_AUDIT, TOP_K_FOR_EMBEDDING_RERANK, LIMIT_CANDIDATES,
)
from ranker import rank_candidates

if __name__ == "__main__":
    rank_candidates(
        jd_path=JD_PATH,
        candidates_path=CANDIDATES_PATH,
        ranked_json_path=RANKED_JSON_PATH,
        final_csv_path=FINAL_CSV_PATH,
        jd_analysis_output_path=JD_ANALYSIS_PATH,
        agent_audit_output_path=AGENT_AUDIT_PATH,
        use_embeddings=USE_LOCAL_EMBEDDINGS,
        use_dev_agents=USE_DEV_AGENTS,
        top_k_embedding=TOP_K_FOR_EMBEDDING_RERANK,
        top_k_agent_audit=TOP_K_FOR_AGENT_AUDIT,
        limit=LIMIT_CANDIDATES,
    )
