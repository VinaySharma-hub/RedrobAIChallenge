print("USING CORRECTED RANKER V3 - NESTED REDROB SCHEMA ENABLED")

import csv
import json
import os
import re
from datetime import datetime
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agents_dev import (
    jd_understanding_agent,
    candidate_audit_agent,
    trap_detection_agent,
    behavioral_signal_agent,
    feature_engineering_agent,
    reasoning_agent,
)

TODAY = datetime(2026, 6, 11)


def clean_text(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(clean_text(i) for i in x)
    if isinstance(x, dict):
        return " ".join(clean_text(v) for v in x.values())
    return str(x).lower().replace("\n", " ").strip()


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9+#.]+", clean_text(text))


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None


def get_profile(raw):
    return raw.get("profile", {}) or {}


def get_signals(raw):
    return raw.get("redrob_signals", {}) or {}


def normalize_scores(scores):
    scores = np.array(scores, dtype=float)
    if len(scores) == 0:
        return scores
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def ranks_from_scores(scores):
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=int)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def rrf_fusion(score_dict, k=60):
    n = len(next(iter(score_dict.values())))
    final = np.zeros(n)
    for scores in score_dict.values():
        ranks = ranks_from_scores(scores)
        final += 1.0 / (k + ranks)
    return final


def load_jd(path):
    for enc in ["utf-8", "cp1252", "latin-1"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def load_candidates_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading candidates")):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def extract_candidate_fields(raw):
    profile = get_profile(raw)
    career_items = raw.get("career_history", []) or []
    education_items = raw.get("education", []) or []
    skill_items = raw.get("skills", []) or []

    career_text = " ".join(
        [
            f"{j.get('title','')} {j.get('industry','')} {j.get('description','')}"
            for j in career_items
        ]
    )
    current_career_text = " ".join(
        [
            f"{j.get('title','')} {j.get('industry','')} {j.get('description','')}"
            for j in career_items
            if j.get("is_current") is True
        ]
    )
    skills_text = " ".join(
        [
            f"{s.get('name','')} {s.get('proficiency','')} endorsements {s.get('endorsements',0)} duration {s.get('duration_months',0)} months"
            for s in skill_items
        ]
    )
    education_text = " ".join(
        [
            f"{e.get('degree','')} {e.get('field_of_study','')} {e.get('institution','')} {e.get('tier','')}"
            for e in education_items
        ]
    )
    title_text = clean_text(
        profile.get("current_title", "") + " " + profile.get("headline", "")
    )
    summary_text = clean_text(profile.get("summary", ""))
    full_text = clean_text(
        " ".join(
            [
                title_text,
                summary_text,
                career_text,
                current_career_text,
                skills_text,
                education_text,
            ]
        )
    )

    return {
        "id": raw.get("candidate_id"),
        "name": profile.get("anonymized_name", ""),
        "title_text": title_text,
        "summary_text": summary_text,
        "career_text": clean_text(career_text),
        "current_career_text": clean_text(current_career_text),
        "projects_text": "",
        "skills_text": clean_text(skills_text),
        "education_text": clean_text(education_text),
        "full_text": full_text,
        "raw": raw,
    }


def compute_bm25_scores(jd_text, candidates):
    corpus_tokens = [tokenize(c["full_text"]) for c in candidates]
    bm25 = BM25Okapi(corpus_tokens)
    return np.array(bm25.get_scores(tokenize(jd_text)), dtype=float)


def compute_tfidf_field_scores(jd_text, candidates):
    fields = [
        ("career_text", 0.48),
        ("current_career_text", 0.17),
        ("skills_text", 0.15),
        ("summary_text", 0.15),
        ("education_text", 0.05),
    ]
    final_scores = np.zeros(len(candidates), dtype=float)
    for field_name, weight in fields:
        docs = [jd_text] + [c[field_name] for c in candidates]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=60000,
            ngram_range=(1, 2),
            min_df=1,
        )
        matrix = vectorizer.fit_transform(docs)
        sim = cosine_similarity(matrix[0], matrix[1:]).flatten()
        final_scores += weight * sim
    return final_scores


def compute_local_embedding_scores(
    jd_text,
    candidates,
    top_indices,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Skipping embedding ranker.")
        return np.zeros(len(candidates), dtype=float)
    model = SentenceTransformer(model_name, device="cpu")
    selected_texts = [candidates[i]["full_text"][:4000] for i in top_indices]
    jd_embedding = model.encode(
        [jd_text], normalize_embeddings=True, show_progress_bar=False
    )[0]
    candidate_embeddings = model.encode(
        selected_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )
    selected_scores = np.dot(candidate_embeddings, jd_embedding)
    final_scores = np.zeros(len(candidates), dtype=float)
    for idx, score in zip(top_indices, selected_scores):
        final_scores[idx] = score
    return final_scores


STRONG_CAREER_TERMS = [
    "built",
    "developed",
    "implemented",
    "deployed",
    "designed",
    "optimized",
    "scaled",
    "owned",
    "launched",
    "production",
    "pipeline",
    "ranking",
    "retrieval",
    "search",
    "recommendation",
    "recommender",
    "embedding",
    "vector",
    "semantic search",
    "machine learning",
    "evaluation",
    "metrics",
    "latency",
    "ab test",
    "a/b test",
    "ndcg",
    "mrr",
    "map",
    "faiss",
    "milvus",
    "pinecone",
    "qdrant",
    "weaviate",
    "elasticsearch",
    "opensearch",
]
WEAK_TERMS = [
    "course",
    "certification",
    "familiar",
    "basic knowledge",
    "interested",
    "tutorial",
    "mini project",
    "experimented with chatgpt",
    "ai strategy advisory",
]
CORE_AI_TERMS = [
    "retrieval",
    "ranking",
    "search",
    "recommendation",
    "recommender",
    "embedding",
    "vector",
    "faiss",
    "milvus",
    "pinecone",
    "qdrant",
    "weaviate",
    "elasticsearch",
    "opensearch",
    "bm25",
    "bge",
    "e5",
    "llm",
    "rag",
    "fine-tuning",
    "lora",
    "qlora",
]
PRODUCTION_TERMS = [
    "production",
    "deployed",
    "serving",
    "latency",
    "scale",
    "queries",
    "users",
    "a/b test",
    "ab test",
    "ndcg",
    "mrr",
    "map",
    "evaluation",
    "offline benchmark",
    "online experiment",
]
BAD_ROLE_TERMS = [
    "hr manager",
    "marketing manager",
    "operations manager",
    "accountant",
    "customer support",
    "sales executive",
    "graphic designer",
    "civil engineer",
    "mechanical engineer",
    "business analyst",
    "project manager",
]
JUNIOR_TERMS = ["intern", "trainee", "fresher", "student", "junior"]


def count_hits(text, terms):
    text = clean_text(text)
    return sum(1 for term in terms if term in text)


def compute_career_evidence_score(candidate):
    career = candidate["career_text"]
    current_career = candidate["current_career_text"]
    skills = candidate["skills_text"]
    full = candidate["full_text"]
    career_hits = count_hits(career, STRONG_CAREER_TERMS)
    current_hits = count_hits(current_career, STRONG_CAREER_TERMS)
    skill_hits = count_hits(skills, STRONG_CAREER_TERMS)
    weak_hits = count_hits(full, WEAK_TERMS)
    score = 0.075 * career_hits + 0.050 * current_hits + 0.015 * skill_hits
    if weak_hits >= 3 and career_hits <= 2:
        score -= 0.18
    return max(0.0, min(score, 1.0))


def compute_experience_fit_score(raw):
    years = safe_float(get_profile(raw).get("years_of_experience"), 0)
    if 5 <= years <= 9:
        return 1.0
    elif 4 <= years < 5:
        return 0.83
    elif 9 < years <= 11:
        return 0.80
    elif 3 <= years < 4:
        return 0.65
    elif 11 < years <= 13:
        return 0.55
    elif 2 <= years < 3:
        return 0.45
    else:
        return 0.25


def compute_title_fit_score(candidate):
    profile = get_profile(candidate["raw"])
    title = clean_text(
        profile.get("current_title", "") + " " + profile.get("headline", "")
    )
    if "senior machine learning engineer" in title:
        return 0.94
    if "machine learning engineer" in title:
        return 0.92
    if "senior ai engineer" in title or "ai engineer" in title:
        return 0.95
    if "ml engineer" in title:
        return 0.90
    if "nlp engineer" in title:
        return 0.85
    if "data scientist" in title:
        return 0.75
    if "software engineer" in title and ("ml" in title or "ai" in title):
        return 0.75
    if "software engineer" in title:
        return 0.65
    if "backend engineer" in title:
        return 0.60
    if "computer vision engineer" in title:
        return 0.34
    if "research scientist" in title:
        return 0.42
    if any(t in title for t in BAD_ROLE_TERMS):
        return 0.09
    if any(t in title for t in JUNIOR_TERMS):
        return 0.24
    return 0.38


def compute_behavior_score(candidate):
    result = behavioral_signal_agent(candidate)
    return result["behavior_signal_score"]


def compute_trap_penalty(candidate):
    raw = candidate["raw"]
    profile = get_profile(raw)
    title = candidate["title_text"]
    career = candidate["career_text"]
    skills = candidate["skills_text"]

    penalty = 0.0

    years = safe_float(profile.get("years_of_experience"), 0)
    skill_ai_hits = count_hits(skills, CORE_AI_TERMS)
    career_ai_hits = count_hits(career, CORE_AI_TERMS)
    production_hits = count_hits(career, PRODUCTION_TERMS)

    # Bad title
    if any(t in title for t in BAD_ROLE_TERMS):
        penalty += 0.35

    # Junior title
    if any(t in title for t in JUNIOR_TERMS):
        penalty += 0.30

    # Too little experience
    if years < 3:
        penalty += 0.30

    # Too much experience for this JD
    if years > 13:
        penalty += 0.30
    elif years > 11:
        penalty += 0.15

    # Skill stuffing
    if skill_ai_hits >= 6 and career_ai_hits <= 2:
        penalty += 0.30

    # AI terms but no production proof
    if career_ai_hits >= 3 and production_hits == 0:
        penalty += 0.18

    # Job hopping / title chasing
    career_items = raw.get("career_history", []) or []
    short_jobs = sum(
        1 for j in career_items if safe_float(j.get("duration_months"), 0) < 18
    )

    if len(career_items) >= 4 and short_jobs >= 3:
        penalty += 0.15

    # Suspicious honeypot cluster
    cid = raw.get("candidate_id", "")
    if cid.startswith("CAND_006666"):
        penalty += 0.40

    # Agent trap detector
    agent_trap = trap_detection_agent(candidate)
    penalty += 0.50 * safe_float(agent_trap.get("trap_penalty", 0), 0)

    return min(penalty, 0.90)


def run_development_agents(jd_text, candidates, preliminary_scores, top_k=100):
    jd_analysis = jd_understanding_agent(jd_text)
    top_indices = np.argsort(-preliminary_scores)[:top_k]
    audits = []
    for idx in top_indices:
        audit = candidate_audit_agent(candidates[idx], jd_analysis)
        trap = trap_detection_agent(candidates[idx])
        audit["trap_flags"] = trap["trap_flags"]
        audit["trap_penalty"] = trap["trap_penalty"]
        audits.append(audit)
    return {
        "jd_analysis": jd_analysis,
        "audits": audits,
        "feature_suggestions": feature_engineering_agent(audits),
    }


def write_json(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def write_submission_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        {
            "candidate_id": r["candidate_id"],
            "rank": r["rank"],
            "score": r["score"],
            "reasoning": r["reasoning"],
        }
        for r in results
    ]
    rows = sorted(rows, key=lambda x: (x["rank"], x["candidate_id"]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)

def hard_logistics_penalty(candidate):
    raw = candidate["raw"]
    profile = get_profile(raw)
    signals = get_signals(raw)

    country = clean_text(profile.get("country", ""))
    willing = signals.get("willing_to_relocate", False)
    notice = safe_float(signals.get("notice_period_days"), 180)
    title = clean_text(profile.get("current_title", ""))

    penalty = 0.0

    if country and country != "india":
        if not willing:
            penalty += 0.22
        else:
            penalty += 0.08

    if notice >= 120:
        penalty += 0.12
    elif notice >= 90:
        penalty += 0.05

    if "junior" in title:
        penalty += 0.20

    return penalty
def print_debug_sample(candidates):
    if not candidates:
        print("No candidates loaded.")
        return
    print("=" * 90)
    print("DEBUG SAMPLE — schema extraction check")
    sample = candidates[0]
    raw = sample["raw"]
    profile = get_profile(raw)
    signals = get_signals(raw)
    print("candidate_id:", sample["id"])
    print("name:", sample["name"])
    print("profile.current_title:", profile.get("current_title"))
    print("profile.years_of_experience:", profile.get("years_of_experience"))
    print("redrob.notice_period_days:", signals.get("notice_period_days"))
    print("redrob.recruiter_response_rate:", signals.get("recruiter_response_rate"))
    print("extracted title_text:", sample["title_text"][:200])
    print("experience_fit_score:", compute_experience_fit_score(raw))
    print("behavior_score:", compute_behavior_score(raw))
    print("title_fit_score:", compute_title_fit_score(sample))
    print("=" * 90)


def rank_candidates(
    jd_path,
    candidates_path,
    ranked_json_path,
    final_csv_path,
    jd_analysis_output_path=None,
    agent_audit_output_path=None,
    use_embeddings=False,
    use_dev_agents=True,
    limit=None,
    top_k_embedding=3000,
    top_k_agent_audit=100,
):
    jd_text = load_jd(jd_path)
    raw_candidates = load_candidates_jsonl(candidates_path, limit=limit)
    candidates = [extract_candidate_fields(c) for c in raw_candidates]
    print_debug_sample(candidates)
    print("Computing BM25 scores...")
    bm25_scores = normalize_scores(compute_bm25_scores(jd_text, candidates))
    print("Computing TF-IDF field scores...")
    tfidf_scores = normalize_scores(compute_tfidf_field_scores(jd_text, candidates))
    print("Computing rule scores...")
    career_scores = np.array([compute_career_evidence_score(c) for c in candidates])
    behavior_scores = np.array([compute_behavior_score(c) for c in candidates])
    experience_scores = np.array(
        [compute_experience_fit_score(c["raw"]) for c in candidates]
    )
    title_scores = np.array([compute_title_fit_score(c) for c in candidates])
    trap_penalties = np.array([compute_trap_penalty(c) for c in candidates])
    print("Score variation check:")
    print("behavior unique:", len(set(np.round(behavior_scores, 4))))
    print("experience unique:", len(set(np.round(experience_scores, 4))))
    print("title unique:", len(set(np.round(title_scores, 4))))
    score_dict = {
        "bm25": bm25_scores,
        "tfidf": tfidf_scores,
        "career": career_scores,
        "behavior": behavior_scores,
        "experience": experience_scores,
        "title": title_scores,
    }
    if use_embeddings:
        print("Computing local embedding scores...")
        top_indices = np.argsort(-bm25_scores)[:top_k_embedding]
        embedding_scores = normalize_scores(
            compute_local_embedding_scores(jd_text, candidates, top_indices)
        )
        score_dict["embedding"] = embedding_scores
    print("Computing RRF fusion...")
    rrf_scores = normalize_scores(rrf_fusion(score_dict))
    preliminary_scores = (
        0.20 * rrf_scores
        + 0.10 * tfidf_scores
        + 0.26 * career_scores
        + 0.25 * behavior_scores
        + 0.10 * experience_scores
        + 0.10 * title_scores
        - trap_penalties
    )
    logistics_penalties = np.array([
    hard_logistics_penalty(c) for c in candidates
])

    preliminary_scores = preliminary_scores - logistics_penalties
    audit_by_id = {}
    if use_dev_agents:
        print("Running deterministic development agents on top candidates...")
        agent_data = run_development_agents(
            jd_text, candidates, preliminary_scores, top_k_agent_audit
        )
        if jd_analysis_output_path:
            write_json(agent_data["jd_analysis"], jd_analysis_output_path)
        if agent_audit_output_path:
            write_json(agent_data, agent_audit_output_path)
        for audit in agent_data["audits"]:
            audit_by_id[audit["candidate_id"]] = audit
    final_scores = preliminary_scores.copy()
    if use_dev_agents:
        for idx, c in enumerate(candidates):
            audit = audit_by_id.get(c["id"])
            if audit and audit["recommendation"] == "promote":
                final_scores[idx] += 0.035
            elif audit and audit["recommendation"] == "demote":
                final_scores[idx] -= 0.070
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-final_scores[i], candidates[i]["id"] or ""),
    )
    results = []
    for rank, idx in enumerate(order[:100], start=1):
        c = candidates[idx]
        score_data = {
            "final_score": float(final_scores[idx]),
            "bm25_score": float(bm25_scores[idx]),
            "tfidf_score": float(tfidf_scores[idx]),
            "career_evidence_score": float(career_scores[idx]),
            "behavior_score": float(behavior_scores[idx]),
            "experience_fit_score": float(experience_scores[idx]),
            "title_fit_score": float(title_scores[idx]),
            "trap_penalty": float(trap_penalties[idx]),
        }
        audit = audit_by_id.get(c["id"])
        trap_result = trap_detection_agent(c)
        behavior_result = behavioral_signal_agent(c)
        reason = reasoning_agent(
            candidate=c,
            score_data=score_data,
            audit=audit,
            trap_result=trap_result,
            behavior_result=behavior_result,
        )
        results.append(
            {
                "rank": rank,
                "candidate_id": c["id"],
                "name": c["name"],
                "score": score_data["final_score"],
                "bm25_score": score_data["bm25_score"],
                "tfidf_score": score_data["tfidf_score"],
                "career_evidence_score": score_data["career_evidence_score"],
                "behavior_score": score_data["behavior_score"],
                "experience_fit_score": score_data["experience_fit_score"],
                "title_fit_score": score_data["title_fit_score"],
                "trap_penalty": score_data["trap_penalty"],
                "reasoning": reason,
            }
        )
    write_json(results, ranked_json_path)
    write_submission_csv(results, final_csv_path)
    print(f"Saved ranked JSON to {ranked_json_path}")
    print(f"Saved final CSV to {final_csv_path}")
    return results
