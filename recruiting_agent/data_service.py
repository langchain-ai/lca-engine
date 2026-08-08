"""Data-access layer for the recruiting assistant.

Thin wrappers over the systems of record: the applicant tracking system (job
postings), the sourcing/HRIS providers (candidate work history, education, and
skills), and the candidate-profile cache. The agent's tools call these functions 
rather than touching storage directly.

The underlying records live in ``recruiting_records``.
"""

from langsmith import traceable

from .recruiting_records import JOB_POSTINGS, CANDIDATES, RECRUITER_IDS

__all__ = [
    "get_job_posting", "get_candidate_record", "get_candidate_record_by_email",
    "fetch_work_history", "fetch_education", "fetch_skills",
    "get_profile_from_db", "save_profile_to_db",
    "get_recruiter",
]

# Built candidate profiles are cached in memory (keyed by candidate_id) so repeat
# lookups within a run are served without rebuilding.
_PROFILES = {}

# ---------------------------------------------------------------------------
# Public data-access functions
# ---------------------------------------------------------------------------
def get_job_posting(job_id):
    "Return the job posting record for job_id from the ATS, or None if not found."
    return JOB_POSTINGS.get(job_id)


def get_candidate_record(candidate_id):
    "Return the source candidate record for candidate_id, or None if not found."
    return CANDIDATES.get(candidate_id)


def get_candidate_record_by_email(email):
    "Return the source candidate record matching email (case-insensitive), or None if not found."
    needle = (email or "").strip().lower()
    if not needle:
        return None
    for record in CANDIDATES.values():
        if (record.get("email") or "").strip().lower() == needle:
            return record
    return None


def get_recruiter(recruiter):
    "Return the recruiter directory record for a recruiter_id or name (case-insensitive), or None if not found."
    needle = (recruiter or "").strip().lower()
    for record in RECRUITER_IDS:
        if needle in (record["recruiter_id"].lower(), record["name"].lower()):
            return record
    return None


@traceable(run_type="tool", name="fetch_work_history")
def fetch_work_history(candidate_id):
    return CANDIDATES[candidate_id]["work_history"]


@traceable(run_type="tool", name="fetch_education")
def fetch_education(candidate_id):
    return CANDIDATES[candidate_id]["education"]


@traceable(run_type="tool", name="fetch_skills")
def fetch_skills(candidate_id):
    return CANDIDATES[candidate_id]["skills"]


@traceable(run_type="tool", name="get_profile_from_db")
def get_profile_from_db(candidate_id):
    "Look up a stored candidate profile. Returns {'candidate_profile': record|None}."
    return {"candidate_profile": _PROFILES.get(candidate_id)}


@traceable(run_type="tool", name="save_profile_to_db")
def save_profile_to_db(candidate_id, profile):
    "Persist a candidate profile to the profile store."
    _PROFILES[candidate_id] = profile
    return {"saved": True}

def add_candidate_skill(candidate_id, skill):
    "Add a skill to a candidate's source-of-truth record."
    record = CANDIDATES.get(candidate_id)
    if record is None:
        return {"updated": False, "found": False}
    skills = list(record["skills"])
    if skill not in skills:
        skills.append(skill)
    return {"updated": True, "found": True, "skills": skills}