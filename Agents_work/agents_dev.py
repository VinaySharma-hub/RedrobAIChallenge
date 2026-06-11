"""
Candidate ranking agents — improved version.

Fixes applied relative to the previous implementation, all traced back to
issues found when auditing two real submissions against the JD,
candidate_schema.json, and redrob_signals_doc:

  1. Honeypot / impossible-profile detection: candidates whose
     profile.years_of_experience does not reconcile with the sum of
     career_history[*].duration_months (the dataset's stated honeypot
     pattern — "subtly impossible profiles").

  2. JD-explicit disqualifiers that were previously invisible to the
     trap detector:
       - "people who have only worked at consulting firms" (TCS, Infosys,
         Wipro, Accenture, Cognizant, Capgemini, ...) for their ENTIRE career.
       - "people whose primary expertise is computer vision, speech, or
         robotics ... without significant NLP/IR exposure".
       - "people whose work has been entirely on closed-source proprietary
         systems for 5+ years without external validation" (approximated
         via career_history industry == research/academia with no
         product-company experience).
       - "title-chasers" — Senior -> Staff -> Principal in <=1.5y hops.

  3. A new behavioral_signal_agent that turns redrob_signals + profile
     location/country/willing_to_relocate into a single availability /
     hireability modifier, explicitly encoding the JD's stated
     preferences:
       - sub-30-day notice preferred, 30-60 acceptable (org can buy out
         up to 30 days), 60-90 marginal, >90 penalized.
       - candidates outside India who are not willing_to_relocate are
         heavily penalized (JD: "we don't sponsor work visas").
       - inactivity (last_active_date), recruiter_response_rate, and
         open_to_work_flag combine into an availability score, matching
         the README's "perfect-on-paper but unreachable" guidance.

  4. reasoning_agent now surfaces ALL of the above as human-readable
     "Concern:" / "Note:" clauses instead of only the single highest
     priority risk flag, so the CSV reasoning column is auditable
     against Stage 5 interview questions.

  5. feature_engineering_agent extended with rules for the new flag
     categories so the suggested-rule loop can actually act on them.

The candidate dict shape is unchanged from the previous version:
    {
        "id": "CAND_XXXXXXX",
        "career_text": "<lowercased concatenation of career_history titles+descriptions>",
        "skills_text": "<lowercased concatenation of skill names>",
        "summary_text": "<lowercased profile.summary>",
        "title_text": "<lowercased profile.current_title>",
        "raw": <full candidate JSON, conforming to candidate_schema.json>,
    }
"""

import re
from datetime import date, datetime
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants derived from the JD's explicit disqualifiers / preferences
# ---------------------------------------------------------------------------

# "People who have only worked at consulting firms ... in their entire career."
CONSULTING_FIRMS = [
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
]

# "People whose primary expertise is computer vision, speech, or robotics
#  without significant NLP/IR exposure."
CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision",
    "image classification",
    "object detection",
    "image segmentation",
    "ocr",
    "speech recognition",
    "asr",
    "tts",
    "robotics",
    "gans",
    "diffusion model",
    "yolo",
]
NLP_IR_TERMS = [
    "nlp",
    "natural language",
    "retrieval",
    "search",
    "ranking",
    "embedding",
    "semantic search",
    "information retrieval",
    "bm25",
    "rag",
    "language model",
    "transformer",
    "bert",
    "llm",
]

# JD: "Pune/Noida-preferred but flexible ... Hyderabad, Pune, Mumbai,
#  Delhi NCR welcome to apply."
PREFERRED_LOCATIONS = [
    "pune",
    "noida",
    "hyderabad",
    "mumbai",
    "delhi",
    "ncr",
    "gurgaon",
    "gurugram",
    "new delhi",
]

# JD: "We'd love sub-30-day notice. We can buy out up to 30 days.
#  30+ day notice candidates are still in scope but the bar gets higher."
NOTICE_SUB_30 = 30
NOTICE_BUYOUT_CEILING = 30
NOTICE_MARGINAL_CEILING = 90

# Used for last_active_date recency calculations. Pass in the actual run
# date if this module is invoked from a long-lived service; the constant
# below matches the dataset's reference "today" for reproducibility.
DEFAULT_TODAY = date(2026, 6, 11)


# ---------------------------------------------------------------------------
# 1. JD understanding agent (unchanged keyword extraction, plus structured
#    disqualifier / preference metadata that the other agents consume)
# ---------------------------------------------------------------------------


def jd_understanding_agent(jd_text: str) -> Dict:
    jd = jd_text.lower()
    keyword_groups = {
        "retrieval": ["retrieval", "search", "information retrieval", "bm25", "tf-idf"],
        "ranking": [
            "ranking",
            "ranker",
            "relevance",
            "search relevance",
            "ndcg",
            "mrr",
            "map",
        ],
        "embeddings": [
            "embedding",
            "embeddings",
            "vector",
            "semantic search",
            "vector database",
        ],
        "vector_db": [
            "faiss",
            "milvus",
            "pinecone",
            "qdrant",
            "weaviate",
            "elasticsearch",
            "opensearch",
        ],
        "ml": ["machine learning", "ml", "model", "training", "evaluation"],
        "python": ["python", "pandas", "numpy", "scikit-learn"],
        "production": [
            "production",
            "deployed",
            "scalable",
            "pipeline",
            "latency",
            "a/b test",
        ],
        "recommendation": ["recommender", "recommendation", "personalization"],
        "llm": ["llm", "rag", "fine-tuning", "lora", "qlora"],
    }
    must_have_keywords = []
    for terms in keyword_groups.values():
        if any(t in jd for t in terms):
            must_have_keywords.extend(terms)

    good_to_have_keywords = []
    if "github" in jd:
        good_to_have_keywords.append("github")
    if "startup" in jd:
        good_to_have_keywords.append("startup")
    if "hr-tech" in jd or "recruiting" in jd:
        good_to_have_keywords.extend(["hr-tech", "recruiting", "marketplace"])

    # --- new: structured disqualifiers, parsed from the JD's own
    # "Things we explicitly do NOT want" / "On location, comp, and
    # logistics" sections. These are deliberately conservative — they only
    # fire when the JD text actually contains the corresponding language,
    # so this agent degrades gracefully if pointed at a different JD.
    disqualifiers = {
        "consulting_only": "consulting firms" in jd or "tcs" in jd or "infosys" in jd,
        "cv_speech_robotics_only": "computer vision" in jd and "robotics" in jd,
        "pure_research_only": "pure research" in jd or "research-only" in jd,
        "title_chasers": "title-chaser" in jd or "title chaser" in jd,
        "no_visa_sponsorship": "don't sponsor" in jd
        or "do not sponsor" in jd
        or "no sponsor" in jd,
    }

    preferences = {
        "yoe_band": (5, 9),
        "notice_sub_30_preferred": True,
        "notice_buyout_ceiling_days": NOTICE_BUYOUT_CEILING,
        "notice_marginal_ceiling_days": NOTICE_MARGINAL_CEILING,
        "preferred_locations": PREFERRED_LOCATIONS,
        "country_required_for_no_relocate": "india",
    }

    return {
        "must_have_keywords": sorted(set(must_have_keywords)),
        "good_to_have_keywords": sorted(set(good_to_have_keywords)),
        "negative_signals": [
            "pure research without production evidence",
            "keyword stuffing",
            "skills listed without career proof",
            "too many unrelated technologies",
            "junior profile for senior JD",
            "course-only experience",
            "LLM/RAG mentioned only as buzzword",
            "AI strategy advisory without hands-on implementation",
        ],
        "strong_evidence_phrases": [
            "built search system",
            "implemented ranking model",
            "deployed ml pipeline",
            "optimized retrieval",
            "worked on recommender system",
            "built vector search",
            "evaluated ranking metrics",
            "production machine learning",
            "owned end-to-end model pipeline",
            "a/b test",
            "ndcg",
            "mrr",
            "map",
            "index refresh",
            "embedding drift",
            "retrieval quality",
        ],
        "weak_evidence_phrases": [
            "completed course",
            "basic knowledge",
            "familiar with",
            "watched tutorials",
            "certified in",
            "interested in ai",
            "worked on mini project",
            "experimented with chatgpt",
            "ai strategy advisory",
        ],
        "disqualifiers": disqualifiers,
        "preferences": preferences,
    }


# ---------------------------------------------------------------------------
# 2. Candidate audit agent (unchanged core logic; kept for compatibility
#    with feature_engineering_agent's "demote" bookkeeping)
# ---------------------------------------------------------------------------


def candidate_audit_agent(candidate: Dict, jd_analysis: Dict) -> Dict:
    career = candidate.get("career_text", "")
    skills = candidate.get("skills_text", "")
    summary = candidate.get("summary_text", "")
    title = candidate.get("title_text", "")
    full = f"{career} {skills} {summary} {title}".lower()

    must_hits, career_hits, skill_only_hits = [], [], []
    for kw in jd_analysis["must_have_keywords"]:
        kw_l = kw.lower()
        if kw_l in full:
            must_hits.append(kw)
        if kw_l in career:
            career_hits.append(kw)
        elif kw_l in skills:
            skill_only_hits.append(kw)

    strong_evidence_hits = [
        p for p in jd_analysis["strong_evidence_phrases"] if p in full
    ]
    weak_evidence_hits = [p for p in jd_analysis["weak_evidence_phrases"] if p in full]

    risk_flags = []
    if len(skill_only_hits) >= 5 and len(career_hits) <= 1:
        risk_flags.append("JD keywords appear mostly in skills, not career history")
    if len(weak_evidence_hits) >= 2:
        risk_flags.append("Weak/course-like evidence detected")
    if any(x in title for x in ["intern", "trainee", "fresher", "student", "junior"]):
        risk_flags.append("Junior title risk")
    if len(set(re.findall(r"[a-zA-Z+#.]+", skills))) > 80:
        risk_flags.append("Very large skill list; possible keyword stuffing")

    if len(career_hits) >= 5 and not risk_flags:
        recommendation = "promote"
    elif len(risk_flags) >= 2:
        recommendation = "demote"
    else:
        recommendation = "keep"

    return {
        "candidate_id": candidate.get("id"),
        "must_hits": must_hits,
        "career_hits": career_hits,
        "skill_only_hits": skill_only_hits,
        "strong_evidence_hits": strong_evidence_hits,
        "weak_evidence_hits": weak_evidence_hits,
        "risk_flags": risk_flags,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Helpers shared by trap_detection_agent and behavioral_signal_agent
# ---------------------------------------------------------------------------


def _career_derived_years(career_history: List[Dict]) -> float:
    """Sum of duration_months across all career_history entries, in years."""
    total_months = sum((r.get("duration_months") or 0) for r in career_history)
    return total_months / 12.0


def _is_consulting_only(career_history: List[Dict]) -> bool:
    if not career_history:
        return False
    companies = [(r.get("company") or "").lower() for r in career_history]
    return all(any(firm in c for firm in CONSULTING_FIRMS) for c in companies)


def _is_cv_speech_robotics_only(career_history: List[Dict]) -> bool:
    """
    True if every role's title+description leans CV/speech/robotics AND
    no role shows meaningful NLP/IR exposure. Mirrors the JD line:
    "People whose primary expertise is computer vision, speech, or
    robotics ... without significant NLP/IR exposure."
    """
    if not career_history:
        return False
    any_cv_role = False
    any_nlp_ir = False
    for r in career_history:
        text = f"{r.get('title','')} {r.get('description','')}".lower()
        if any(t in text for t in CV_SPEECH_ROBOTICS_TERMS):
            any_cv_role = True
        if any(t in text for t in NLP_IR_TERMS):
            any_nlp_ir = True
    return any_cv_role and not any_nlp_ir


def _is_pure_research_no_production(career_history: List[Dict]) -> bool:
    """
    Approximates: "career in pure research environments (academic labs,
    research-only roles) without any production deployment."
    Heuristic: every role's industry mentions research/academia AND no
    role's description contains a production/deployment signal.
    """
    if not career_history:
        return False
    industries = [(r.get("industry") or "").lower() for r in career_history]
    if not all(("research" in i) or ("academ" in i) for i in industries):
        return False
    production_terms = [
        "production",
        "deployed",
        "shipped",
        "live",
        "scaled",
        "users",
        "latency",
    ]
    any_production = any(
        any(t in (r.get("description") or "").lower() for t in production_terms)
        for r in career_history
    )
    return not any_production


def _is_title_chaser(career_history: List[Dict]) -> bool:
    """
    JD: "career trajectory shows you optimizing for Senior -> Staff ->
    Principal titles by switching companies every 1.5 years."
    Heuristic: 3+ roles, each <=18 months, with strictly escalating
    seniority keywords across different companies.
    """
    if len(career_history) < 3:
        return False
    seniority_order = [
        "junior",
        "associate",
        "",
        "senior",
        "staff",
        "lead",
        "principal",
        "director",
    ]

    def seniority_rank(title: str) -> int:
        title_l = title.lower()
        for idx, kw in enumerate(seniority_order):
            if kw and kw in title_l:
                return idx
        return 2  # mid-level default

    short_stints = sum(
        1 for r in career_history if (r.get("duration_months") or 999) <= 18
    )
    if short_stints < 3:
        return False

    ranks = [seniority_rank(r.get("title", "")) for r in career_history]
    # career_history is typically ordered most-recent-first; check for a
    # monotonically non-decreasing seniority trend reading oldest -> newest
    ranks_chrono = list(reversed(ranks))
    return (
        all(b >= a for a, b in zip(ranks_chrono, ranks_chrono[1:]))
        and ranks_chrono[-1] > ranks_chrono[0]
    )


# ---------------------------------------------------------------------------
# 3. Trap detection agent — extended with honeypot + JD disqualifier checks
# ---------------------------------------------------------------------------


def trap_detection_agent(candidate: Dict) -> Dict:
    raw = candidate.get("raw", {}) or {}
    profile = raw.get("profile", {}) or {}
    career_history = raw.get("career_history", []) or []

    career = candidate.get("career_text", "")
    skills = candidate.get("skills_text", "")
    title = candidate.get("title_text", "")
    full = f"{career} {skills} {title}".lower()

    flags: List[str] = []
    penalty = 0.0

    # --- existing keyword-stuffing checks -----------------------------
    skill_tokens = set(re.findall(r"[a-zA-Z+#.]+", skills))
    if len(skill_tokens) > 100:
        flags.append("Extreme skill count")
        penalty += 0.25
    elif len(skill_tokens) > 70:
        flags.append("High skill count")
        penalty += 0.15

    ai_terms = [
        "llm",
        "rag",
        "openai",
        "langchain",
        "vector database",
        "transformer",
        "deep learning",
        "nlp",
        "computer vision",
        "ranking",
        "retrieval",
        "recommendation",
        "embedding",
        "faiss",
        "milvus",
        "pinecone",
        "qdrant",
        "weaviate",
    ]
    skill_ai_hits = sum(1 for t in ai_terms if t in skills)
    career_ai_hits = sum(1 for t in ai_terms if t in career)
    if skill_ai_hits >= 6 and career_ai_hits <= 1:
        flags.append("AI buzzwords mostly in skills, not work history")
        penalty += 0.25

    if any(t in title for t in ["intern", "student", "fresher"]) and skill_ai_hits >= 8:
        flags.append("Junior profile with excessive expert-like skills")
        penalty += 0.20

    if "expert" in full and len(skill_tokens) > 60:
        flags.append("Expert wording with broad skill list")
        penalty += 0.10

    # --- NEW: honeypot — stated YoE vs career-history-derived YoE -----
    stated_yoe = profile.get("years_of_experience")
    derived_yoe = _career_derived_years(career_history)
    if stated_yoe is not None and derived_yoe > 0:
        diff = abs(stated_yoe - derived_yoe)
        if diff > 5:
            flags.append(
                f"Stated experience ({stated_yoe}y) is wildly inconsistent with "
                f"career history ({derived_yoe:.1f}y) — likely honeypot"
            )
            penalty += 0.40
        elif diff > 2:
            flags.append(
                f"Stated experience ({stated_yoe}y) does not reconcile with "
                f"career history ({derived_yoe:.1f}y)"
            )
            penalty += 0.20

    # --- NEW: JD-explicit disqualifiers --------------------------------
    if _is_consulting_only(career_history):
        flags.append(
            "Entire career at consulting/services firms (TCS/Infosys/Wipro/etc.)"
        )
        penalty += 0.30

    if _is_cv_speech_robotics_only(career_history):
        flags.append("Primary expertise is CV/speech/robotics with no NLP/IR exposure")
        penalty += 0.35

    if _is_pure_research_no_production(career_history):
        flags.append(
            "Pure research/academic background with no production deployment evidence"
        )
        penalty += 0.30

    if _is_title_chaser(career_history):
        flags.append(
            "Career pattern resembles title-chasing (rapid Senior->Staff->Principal hops)"
        )
        penalty += 0.15

    return {"trap_flags": flags, "trap_penalty": min(penalty, 0.70)}


# ---------------------------------------------------------------------------
# 4. NEW — behavioral_signal_agent
#    Converts redrob_signals + profile location/relocation into a single
#    availability/hireability modifier and human-readable flags.
# ---------------------------------------------------------------------------


def behavioral_signal_agent(candidate: Dict, today: Optional[date] = None) -> Dict:
    today = today or DEFAULT_TODAY
    raw = candidate.get("raw", {}) or {}
    profile = raw.get("profile", {}) or {}
    signals = raw.get("redrob_signals", {}) or {}

    score = 0.5  # neutral baseline; clamped to [0, 1] at the end
    flags: List[str] = []

    # --- availability / engagement ------------------------------------
    open_to_work = signals.get("open_to_work_flag", False)
    if not open_to_work:
        score -= 0.15
        flags.append("Not marked open to work")

    last_active = signals.get("last_active_date")
    days_inactive = None
    if last_active:
        try:
            days_inactive = (
                today - datetime.strptime(last_active, "%Y-%m-%d").date()
            ).days
        except ValueError:
            days_inactive = None
    if days_inactive is not None:
        if days_inactive > 180:
            score -= 0.20
            flags.append(f"Inactive for {days_inactive} days (>6 months)")
        elif days_inactive > 90:
            score -= 0.10
            flags.append(f"Inactive for {days_inactive} days")
        elif days_inactive <= 30:
            score += 0.10

    rr = signals.get("recruiter_response_rate")
    if rr is not None:
        score += (rr - 0.5) * 0.3
        if rr < 0.2 and not open_to_work:
            flags.append("Very low recruiter response rate and not open to work")
            score -= 0.10

    # --- notice period, per JD's stated policy -------------------------
    notice = signals.get("notice_period_days")
    if notice is not None:
        if notice <= 15:
            score += 0.18
        elif notice <= 30:
            score += 0.14
        elif notice <= 60:
            score += 0.04
        elif notice <= 90:
            score -= 0.08
            flags.append(f"Notice period {notice}d is above ideal range")
        else:
            score -= 0.25
            flags.append(f"Notice period {notice}d is well above JD preference")
    # --- location / relocation / visa sponsorship -----------------------
    country = (profile.get("country") or "").lower()
    location = (profile.get("location") or "").lower()
    willing_relocate = signals.get("willing_to_relocate", False)

    if country and country != "india":
        if not willing_relocate:
            score -= 0.65
            flags.append(
                f"Based in {profile.get('country')}, not willing to relocate — JD does not sponsor work visas"
            )
        else:
            score -= 0.40
            flags.append(f"Based in {profile.get('country')} but open to relocation")
    else:
        if not any(loc in location for loc in PREFERRED_LOCATIONS):
            score -= 0.05
            flags.append(
                "Located in India but outside JD-preferred cities (Pune/Noida/NCR/Hyderabad/Mumbai)"
            )

    # --- interview/offer history (cheap signal of "actually hireable") --
    interview_rate = signals.get("interview_completion_rate")
    if interview_rate is not None and interview_rate < 0.4:
        score -= 0.05
        flags.append(f"Low interview completion rate ({interview_rate:.0%})")

    return {
        "behavior_signal_score": max(0.0, min(1.0, score)),
        "behavior_flags": flags,
    }


# ---------------------------------------------------------------------------
# 5. Feature engineering agent — extended with rules for the new flag types
# ---------------------------------------------------------------------------


def feature_engineering_agent(
    audits: List[Dict], trap_results: Optional[List[Dict]] = None
) -> Dict:
    suggested_rules = []
    demoted = [a for a in audits if a["recommendation"] == "demote"]

    skill_only_problem = sum(
        1
        for a in demoted
        if len(a.get("skill_only_hits", [])) >= 5 and len(a.get("career_hits", [])) <= 1
    )
    junior_problem = sum(
        1 for a in demoted if any("Junior" in flag for flag in a.get("risk_flags", []))
    )

    if skill_only_problem >= 3:
        suggested_rules.append(
            {
                "rule": "penalize_skill_only_match",
                "logic": "If many JD keywords appear in skills but not career history, add trap penalty.",
            }
        )
    if junior_problem >= 2:
        suggested_rules.append(
            {
                "rule": "penalize_junior_title",
                "logic": "If title contains intern/fresher/student and JD requires experience, demote.",
            }
        )

    suggested_rules.append(
        {
            "rule": "career_evidence_boost",
            "logic": "Boost candidates with JD keywords in work history more than skills.",
        }
    )

    # --- NEW: rules driven by trap_detection_agent output ---------------
    if trap_results:
        honeypot_count = sum(
            1
            for t in trap_results
            if any(
                "honeypot" in f.lower() or "inconsistent" in f.lower()
                for f in t.get("trap_flags", [])
            )
        )
        consulting_count = sum(
            1
            for t in trap_results
            if any("consulting" in f.lower() for f in t.get("trap_flags", []))
        )
        cv_only_count = sum(
            1
            for t in trap_results
            if any("cv/speech/robotics" in f.lower() for f in t.get("trap_flags", []))
        )
        research_count = sum(
            1
            for t in trap_results
            if any("pure research" in f.lower() for f in t.get("trap_flags", []))
        )

        if honeypot_count >= 1:
            suggested_rules.append(
                {
                    "rule": "exclude_yoe_career_mismatch",
                    "logic": (
                        "If |stated years_of_experience - sum(career_history.duration_months)/12| > 5, "
                        "exclude the candidate from top-100 entirely (treat as honeypot). "
                        "If the gap is between 2 and 5 years, apply a 0.20 score penalty."
                    ),
                }
            )
        if consulting_count >= 1:
            suggested_rules.append(
                {
                    "rule": "exclude_consulting_only_career",
                    "logic": (
                        "If every career_history.company matches a known consulting/services "
                        "firm (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini), exclude "
                        "the candidate per JD's explicit 'consulting firms' disqualifier."
                    ),
                }
            )
        if cv_only_count >= 1:
            suggested_rules.append(
                {
                    "rule": "exclude_cv_speech_robotics_only",
                    "logic": (
                        "If career_history shows CV/speech/robotics roles with no NLP/IR "
                        "terms anywhere in titles or descriptions, exclude per JD's "
                        "'primary expertise is computer vision, speech, or robotics' disqualifier."
                    ),
                }
            )
        if research_count >= 1:
            suggested_rules.append(
                {
                    "rule": "demote_pure_research_no_production",
                    "logic": (
                        "If every career_history entry is in a research/academic industry "
                        "and no description mentions production/deployment, apply a 0.25-0.30 "
                        "penalty per JD's 'pure research environments without production "
                        "deployment' disqualifier."
                    ),
                }
            )

    suggested_rules.append(
        {
            "rule": "behavioral_signal_modifier",
            "logic": (
                "Multiply the base relevance score by behavioral_signal_score from "
                "behavioral_signal_agent (range ~0.0-1.0, baseline 0.5), so that "
                "open_to_work, last_active_date recency, recruiter_response_rate, "
                "notice_period_days, and location/relocation/visa fit all influence "
                "final ranking — not just skill/title relevance."
            ),
        }
    )

    return {"suggested_rules": suggested_rules}


# ---------------------------------------------------------------------------
# 6. Reasoning agent — now surfaces honeypot/disqualifier/behavioral flags
# ---------------------------------------------------------------------------


def reasoning_agent(
    candidate: Dict,
    score_data: Dict,
    audit: Optional[Dict] = None,
    trap_result: Optional[Dict] = None,
    behavior_result: Optional[Dict] = None,
) -> str:
    raw = candidate.get("raw", {}) or {}
    profile = raw.get("profile", {}) or {}
    signals = raw.get("redrob_signals", {}) or {}

    title = profile.get("current_title", "Candidate")
    years = profile.get("years_of_experience", None)
    location = profile.get("location", "")
    country = profile.get("country", "")

    parts: List[str] = []

    if years is not None:
        parts.append(f"{title} with {years} years of experience.")

    if score_data.get("career_evidence_score", 0) >= 0.70:
        parts.append(
            "Career history shows strong evidence for AI/ML retrieval or ranking work."
        )
    elif score_data.get("career_evidence_score", 0) >= 0.40:
        parts.append(
            "Career history has some relevant AI/ML evidence, though not uniformly strong."
        )

    if score_data.get("title_fit_score", 0) >= 0.85:
        parts.append(
            "Current title is closely aligned with the Senior AI Engineer role."
        )
    elif score_data.get("title_fit_score", 0) <= 0.20:
        parts.append(
            "Current title is not directly aligned, so the profile is penalized."
        )

    if score_data.get("behavior_score", 0) >= 0.45:
        notice = signals.get("notice_period_days", "unknown")
        response = signals.get("recruiter_response_rate", "unknown")
        parts.append(
            f"Hiring signals are usable: notice period {notice} days and recruiter response rate {response}."
        )

    if audit:
        if audit.get("career_hits"):
            parts.append(
                "Relevant career-history terms include "
                + ", ".join(audit["career_hits"][:4])
                + "."
            )
        # Surface ALL audit risk flags, not just the first.
        for flag in audit.get("risk_flags", []):
            parts.append(f"Concern: {flag}.")

    # --- NEW: honeypot / JD-disqualifier flags from trap_detection_agent
    if trap_result:
        for flag in trap_result.get("trap_flags", []):
            # Skip generic skill-stuffing flags here if already covered by audit
            if flag not in (audit or {}).get("risk_flags", []):
                parts.append(f"Concern: {flag}.")

    # --- NEW: availability / location / notice flags from behavioral agent
    if behavior_result:
        for flag in behavior_result.get("behavior_flags", []):
            parts.append(f"Note: {flag}.")

    if location or country:
        parts.append(f"Location: {location}, {country}.")

    if not parts:
        parts.append(
            "Selected by combined retrieval, field similarity, career evidence, behavioral signals, and trap checks."
        )

    return " ".join(parts)[:1200]
