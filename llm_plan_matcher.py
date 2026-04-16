"""
plan_optimizer.py
=================
Single-file bulk insurance plan matcher.
  - Reads patients_results.xlsx  (sheet: "pVerify Results")
  - For every "Processed" row, attempts DB exact-match first,
    then falls back to LLM-assisted SQL discovery + LLM final ranking.
  - Writes results to potential_plan_matches_llm.xlsx
  - Writes a detailed run log to plan_optimizer_<timestamp>.log

Usage:
    python plan_optimizer.py
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import os
import re
import json
import logging
import datetime
import traceback
from typing import Dict, List, Optional, Any, Union

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import boto3
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────
LOG_TIMESTAMP  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE       = f"plan_optimizer_test_{LOG_TIMESTAMP}.log"
# Load filenames from environment, with defaults
INPUT_FILE     = os.getenv("EXCEL_FILE_PATH", "patients_results.xlsx")
OUTPUT_FILE    = os.getenv("OUTPUT_FILE_PATH", "potential_plan_matches_test_llm.xlsx")
SHEET_NAME     = os.getenv("EXCEL_SHEET_NAME", "pVerify Results")

def setup_logging() -> logging.Logger:
    """
    Configure the root logger to output to both the console and a timestamped file.
    
    Returns:
        logging.Logger: The configured logger instance.
    """
    log_format = "%(asctime)s  [%(levelname)-8s]  %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("plan_optimizer")
    logger.setLevel(logging.DEBUG)

    # Console handler (INFO and above)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # File handler (DEBUG and above – captures detailed execution)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

log = setup_logging()

# US State Mapping for strict regional filtering
US_STATE_MAP = {
    'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DC': 'District of Columbia', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
    'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MO': 'Missouri', 'MS': 'Mississippi', 'MT': 'Montana', 'NC': 'North Carolina',
    'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia',
    'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'
}

def get_state_variants(state_input: str) -> List[str]:
    """Returns both abbreviation and full name for a given state input."""
    if not state_input: return []
    s = state_input.strip().upper()
    variants = [s]
    # If it's an abbreviation, add full name
    if s in US_STATE_MAP:
        variants.append(US_STATE_MAP[s])
    # If it's a full name, add abbreviation
    else:
        for abbr, full in US_STATE_MAP.items():
            if s.lower() == full.lower():
                variants.append(abbr)
                break
    return list(set(variants))


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
# Database Connection Details
DB_HOST      = os.getenv("DB_HOST")
DB_NAME      = os.getenv("DB_NAME")
DB_USER      = os.getenv("DB_USER")
DB_PASSWORD  = os.getenv("DB_PASSWORD")
DB_PORT      = os.getenv("DB_PORT")

# Bedrock Configuration
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.getenv("AWS_BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID      = "meta.llama3-8b-instruct-v1:0"

# Matching Logic Thresholds
CONFIDENCE_THRESHOLD = 0.5   # Minimum score required to accept an LLM match as definitive

def call_llm(prompt: str) -> str:
    """
    Unified helper to call AWS Bedrock Llama 3.
    """
    try:
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]

        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=messages,
            inferenceConfig={
                "maxTokens": 2048,
                "temperature": 0.0
            }
        )

        return response["output"]["message"]["content"][0]["text"]

    except Exception as e:
        log.error(f"Bedrock API call failed: {e}")
        raise e


# ─────────────────────────────────────────────────────────────
# SECURITY & HELPERS
# ─────────────────────────────────────────────────────────────
def safe_json_parse(text: str) -> Dict:
    """
    Robustly extracts and parses JSON from a string, even if it contains 
    markdown fences or conversational filler.
    """
    if not text:
        return {}
    
    # Try direct parse first
    try:
        return json.loads(text.strip(), strict=False)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown blocks
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip(), strict=False)
        except json.JSONDecodeError:
            pass

    # Try finding the first '{' and last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1], strict=False)
        except json.JSONDecodeError:
            pass
            
    return {}


def validate_sql_safety(sql: str) -> bool:
    """
    Strictly validates that a generated SQL string is a safe SELECT statement 
    targeting only the authorized plan_details table.

    Args:
        sql (str): The raw SQL query string to validate.

    Returns:
        bool: True if the query is deemed safe, False otherwise.
    """
    if not sql:
        return False

    sql_clean = sql.strip()
    sql_upper = sql_clean.upper()

    # 1. Primary Authorization: Must be a SELECT statement
    if not sql_upper.startswith("SELECT"):
        log.error("SQL Security Block: Query must start with 'SELECT'.")
        return False

    # 2. Injection Prevention: Disallow multiple statements (internal semicolons)
    # We strip the trailing semicolon if it exists to allow standard SQL termination
    core_sql = sql_clean.rstrip().rstrip(';')
    if ";" in core_sql:
        log.error("SQL Security Block: Internal semicolon detected. Multiple statements are forbidden.")
        return False

    # 3. Modification Prevention: Blacklist destructive keywords
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER", "GRANT", "REVOKE", "COMMENT", "RENAME"]
    for word in forbidden:
        pattern = rf"\b{word}\b"
        if re.search(pattern, sql_upper):
            log.error(f"SQL Security Block: Prohibited keyword '{word}' detected.")
            return False

    # 4. Scope Restriction: Ensure it only queries the allowed table
    target_table = "ebv_genai.plan_details"
    if target_table.upper() not in sql_upper:
        log.error(f"SQL Security Block: Query must target authorized table '{target_table}'.")
        return False

    return True


def json_serial(obj: Any) -> Union[str, Any]:
    """
    JSON serializer for objects not serializable by default json code (e.g., datetime).

    Args:
        obj (Any): The object to serialize.

    Returns:
        Union[str, Any]: ISO format string for dates/times, or the original object.
    """
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def get_db_connection() -> Any:
    """
    Establishes and returns a live connection to the PostgreSQL database.

    Returns:
        psycopg2.extensions.connection: A database connection object.

    Raises:
        psycopg2.Error: If the connection cannot be established.
    """
    log.debug(f"Connecting to DB: host={DB_HOST}, port={DB_PORT}, db={DB_NAME}")
    try:
        conn = psycopg2.connect(
            host=DB_HOST, 
            port=DB_PORT, 
            database=DB_NAME,
            user=DB_USER, 
            password=DB_PASSWORD, 
            connect_timeout=15,
        )
        log.debug("DB connection established successfully.")
        return conn
    except Exception as exc:
        log.error(f"Database connection failed: {exc}")
        raise


def infer_subtype(policy_type: str, plan_name: str) -> Optional[str]:
    """
    Infers the insurance plan subtype (PPO, HMO, POS, etc.) from the policy 
    type field or keywords within the plan name.

    Args:
        policy_type (str): The raw policy type from the Excel file.
        plan_name (str): The descriptive name of the plan.

    Returns:
        Optional[str]: The inferred subtype or None if not found.
    """
    if policy_type and policy_type.strip():
        return policy_type.strip()
        
    upper_plan = plan_name.upper()
    for keyword in ("PPO", "HMO", "POS", "EPO", "HDHP"):
        if keyword in upper_plan:
            log.debug(f"Inferred subtype '{keyword}' from plan name '{plan_name}'")
            return keyword
    return None


# ─────────────────────────────────────────────────────────────
# PHASE 1 – EXACT DB MATCH
# ─────────────────────────────────────────────────────────────
def find_exact_match(cursor: Any, payer_info: List[str], plan_name: str,
                     plan_sub_type: Optional[str], state_name: Optional[str]) -> Optional[Dict]:
    """
    Attempts to find a direct match in the database using exact string 
    matching or ILIKE patterns for the plan name and payer variants.

    Args:
        cursor (Any): The database cursor to execute queries.
        payer_info (List[str]): A list of payer name variants to search for.
        plan_name (str): The name of the insurance plan.
        plan_sub_type (Optional[str]): The subtype (PPO, HMO, etc.) if known.
        state_name (Optional[str]): The US state name if known.

    Returns:
        Optional[Dict]: The matched plan record as a dictionary, or None.
    """
    if not plan_name:
        log.debug("Exact match skipped: plan_name is empty.")
        return None

    # OR-expand conditions across all payer variants for flexibility
    payer_conditions = " OR ".join(
        [f"LOWER(payer_name) = LOWER(%s)" for _ in payer_info] +
        [f"LOWER(payer_name) LIKE LOWER(%s)" for _ in payer_info]
    )
    params: List = [plan_name] + payer_info + [f"%{p}%" for p in payer_info]

    query = f"""
        SELECT plan_id, plan_name, payer_name, payer_id, plan_type, plan_sub_type, state_name
        FROM ebv_genai.plan_details
        WHERE LOWER(plan_name) = LOWER(%s)
          AND ({payer_conditions})
    """

    if state_name:
        query += " AND LOWER(state_name) = LOWER(%s)"
        params.append(state_name)
    if plan_sub_type:
        query += " AND LOWER(plan_sub_type) = LOWER(%s)"
        params.append(plan_sub_type)

    log.debug(f"Exact-match SQL: plan='{plan_name}', payers={payer_info}")
    cursor.execute(query, params)
    row = cursor.fetchone()
    
    if row:
        log.debug(f"Exact match found: plan_id={row['plan_id']}")
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────
# PHASE 2 – LLM-GENERATED DISCOVERY SQL
# ─────────────────────────────────────────────────────────────
def generate_discovery_sql(request_data: Dict) -> str:
    """
    Leverages Gemini LLM to interpret insurance plan data and generate a 
    discovery SQL query to find potential matches in the database.

    Args:
        request_data (Dict): A dictionary containing plan_name, payer_info, etc.

    Returns:
        str: The generated SELECT query, or an empty string on failure.
    """
    log.info("Phase 2 – Requesting LLM-assisted discovery SQL...")
    prompt = f"""
You are a PostgreSQL expert and an experienced US health insurance Benefit Verification (BV) agent.
You have deep knowledge of how insurance entities are structured (parent companies, DBAs,
Blue Cross Blue Shield regional licenses, mergers, etc.).

Inputs:
- Payer Info (list of known payer name variants): {request_data.get('payer_info')}
- Plan Name:   {request_data.get('plan_name')}
- Plan Sub Type: {request_data.get('plan_sub_type')}
- State Name:  {request_data.get('state_name')}

Table: ebv_genai.plan_details
Columns: plan_id, plan_name, payer_name, payer_id, plan_type, plan_sub_type, state_name

Goal: Generate a SELECT query returning the TOP 100 most likely candidate rows.

BV Agent Rules for SQL Generation:
1. BROAD DISCOVERY Logic (CRITICAL): You must surface candidates from the requested payer family BROADLY across the state.
   - Goal: Ensure you return up to 100 diverse candidates from both the Primary Payer AND its known Partners.
   - Primary Payer (e.g., Aetna): Search for the Payer and the state, using core Plan Name keywords but keeping the filter loose enough to find 50+ candidates from this payer alone.
   - Partner Payer (e.g., Coventry): Search for the Partner BROADLY in the state (include all results for this partner up to the limit).
   - STRUCTURE: ( (payer_name ILIKE '%Payer1%' AND [loose plan keywords]) OR (payer_name ILIKE '%Partner1%') ) AND (state_name filter)
2. MANDATORY State Filter: You MUST restrict results to the requested 'state_name'. (e.g. state_name = 'PA' OR state_name ILIKE '%Pennsylvania%').
3. MANDATORY Plan Search: Return the query on a SINGLE LINE (no newlines).
4. SQL STRUCTURE (CRITICAL): Your query MUST follow this template to ensure strict isolation:
   SELECT ... FROM ebv_genai.plan_details WHERE ( [Payer/Plan Logic] ) AND ( state_name = '...' OR state_name ILIKE '%...%' )
   Note: The parentheses around the entire Payer/Plan block are MANDATORY to prevent regional results from leaking other payers.
5. Partner Knowledge (STRICT ISOLATION): 
   - 'Cigna' partners ONLY with 'HealthSpring'.
   - 'Highmark' partners ONLY with 'BCBS Pennsylvania'.
   - 'Aetna' partners ONLY with 'Coventry' and 'Meritain Health'.
   - DO NOT mix these. If input is 'Aetna', DO NOT include 'HealthSpring' or 'Highmark' in your SQL.
5. Logic Priority: ALWAYS use parentheses as shown in the DUAL-FILTER example.
5. Column Flexibility: Keywords like 'PPO', 'HMO' can appear in plan_name or plan_sub_type.
6. Limit: Return up to 100 rows.
7. STRICT REGIONAL ISOLATION: DO NOT return plans from states other than {request_data.get('state_name') or 'ALL'}.
8. DO NOT include an ORDER BY clause. Just return the SELECT ... WHERE ... LIMIT 100.

Return ONLY valid JSON – no markdown, no commentary:
{{ "interpreted_state": "<state or null>", "sql_query": "SELECT ... LIMIT 100" }}
"""

    try:
        text = call_llm(prompt)
        log.debug(f"Raw discovery response: {text}")
        parsed = safe_json_parse(text)
        
        sql    = parsed.get("sql_query", "").strip()
        
        # ── SQL Sanitization (Robust Logic) ──
        # 1. Strip ORDER BY entirely – Phase 3 (LLM Ranking) handles scoring.
        #    The LLM frequently generates broken ORDER BY clauses with malformed ::int casts.
        sql = re.sub(r"\s+ORDER\s+BY\s+.*?(?=\s+LIMIT\b)", " ", sql, flags=re.IGNORECASE)
        
        # 2. Parenthesis Balancing: Close any unclosed parentheses
        open_count = sql.count('(')
        close_count = sql.count(')')
        if open_count > close_count:
            if "LIMIT" in sql.upper():
                parts = sql.split("LIMIT")
                balanced_main = parts[0].strip() + (")" * (open_count - close_count))
                sql = f"{balanced_main} LIMIT {parts[1]}"
            else:
                sql += (")" * (open_count - close_count))
        
        state  = parsed.get("interpreted_state")

        if state:
            log.info(f"LLM interpreted state: '{state}'")
        return sql

    except Exception as exc:
        err_msg = str(exc)
        if "Throttling" in err_msg:
            raise RuntimeError("Bedrock API Throttled. Please try again in a few moments.")
        log.error(f"LLM discovery SQL generation failed: {err_msg}")
        return ""


# ─────────────────────────────────────────────────────────────
# PHASE 3 – LLM FINAL RANKING
# ─────────────────────────────────────────────────────────────
def simple_ranking_fallback(request_data: Dict, candidates: List[Dict]) -> List[Dict]:
    """Deterministic scoring fallback if LLM ranking fails."""
    scored = []
    inp_plan = request_data.get("plan_name", "").lower()
    inp_payer = str(request_data.get("payer_info", [])).lower()
    inp_state = request_data.get("state_name", "").lower()
    
    for c in candidates:
        score_val = 0.05 # Baseline
        c_plan = c.get("plan_name", "").lower()
        c_payer = c.get("payer_name", "").lower()
        c_state = c.get("state_name", "").lower()
        
        # State match (Mandatory if input state provided)
        if inp_state:
            if c_state != inp_state:
                scored.append({"plan_id": c["plan_id"], "confidence_score": 0.0})
                continue
            score_val += 0.2
            
        # Payer overlap
        if c_payer in inp_payer or any(p.lower() in c_payer for p in request_data.get("payer_info", [])):
            score_val += 0.2
            
        # Plan name overlap (fuzzy)
        words = [w for w in inp_plan.split() if len(w) > 3]
        for w in words:
            if w in c_plan:
                score_val += (0.5 / len(words)) if words else 0.5
        
        scored.append({"plan_id": c["plan_id"], "confidence_score": min(score_val, 0.95)})
    return scored

def get_llm_final_match(request_data: Dict, candidates: List[Dict]) -> Dict:
    """
    Presents a list of database candidates to Gemini LLM for final 
    selection and confidence scoring based on BV agent logic.

    Args:
        request_data (Dict): The original plan request data.
        candidates (List[Dict]): A list of candidate plan records from the DB.

    Returns:
        Dict: A dictionary containing the best match, confidence score, and reasoning.
    """
    if not candidates:
        log.info("No candidates available for ranking.")
        return {"match": None, "confidence_score": 0.0, "reasoning": "No candidates found."}

    log.info(f"Phase 3 – LLM ranking {len(candidates)} candidate(s)...")

    prompt = f"""
You are an experienced US health insurance Benefit Verification (BV) agent.
Match the original request to the single best database record from the candidates below.

Original Request:
- Payer variants: {request_data.get('payer_info')}
- Plan Name:      {request_data.get('plan_name')}
- Subtype:        {request_data.get('plan_sub_type')}
- Group Name:     {request_data.get('group_name')}
- State:          {request_data.get('state_name')}

Candidate Plans:
{json.dumps(candidates, indent=2, default=json_serial)}

BV Agent Rules:
1. Goal: You MUST evaluate and return a score for EVERY SINGLE candidate plan provided in the list above.
2. No Omissions: Even if the match is very poor, you MUST include its 'plan_id' in the JSON output.
3. Match Logic: 'Highmark BCBS' matches 'BCBS of Pennsylvania'. 'Cigna' is linked with 'HealthSpring'. 'Aetna' is linked with 'Coventry' and 'Meritain Health'.
4. Scoring Guidelines:
- 1.0: Perfect match (All fields match exactly).
- 0.8 - 0.9: Very strong match (Plan name matches, but maybe minor payer variant like 'Aetna' vs 'Aetna Life').
- 0.5 - 0.7: Strong match (Same Payer and similar Plan, but state might be 'National' vs specific state).
- 0.2 - 0.4: Weak match (Matches on Payer OR State, but Plan name is unrelated).
- 0.0: Totally unrelated OR DEALBREAKER (e.g., matching a Commercial plan to a Medicare/Medicaid plan).

DEALBREAKERS (Score = 0.0):
- If the plan name contains 'Medicare' or 'Medicaid' but the user input DOES NOT (and vice versa).
- If the plan types are fundamentally different (e.g. 'PPO' vs 'POS' vs 'HMO') and no better candidates exist.

AI Mandatory Checklist:
1. All Data Evaluation: Review every field (Plan Name, Payer, State) for each candidate.
2. Input Alignment: Compare candidates strictly against the user's specific input fields.
3. Partner Payer Logic: Prioritize known partners (Cigna/HealthSpring or Highmark/BCBS) as nearly equal to direct matches. These MUST BE SCORED positively.
4. Complete Ranking: You MUST provide a 'confidence_score' for EVERY candidate provided in the list below. DO NOT skip any.
5. Correct Ranking: Ensure the final JSON list is sorted from highest confidence to lowest.

Return ONLY valid JSON. YOUR RESPONSE MUST BE A SINGLE JSON OBJECT.
DO NOT WRITE PYTHON CODE. DO NOT WRITE ANY COMMENTARY. 
REQUIRED FORMAT EXAMPLE:
{{
  "scored_candidates": [
    {{ "plan_id": "...", "confidence_score": 0.95 }}
  ]
}}

YOUR JSON:
"""

    try:
        text = call_llm(prompt)
        log.debug(f"Raw ranking response: {text}")
        result = safe_json_parse(text)
        if not result:
            log.warning("AI ranking returned invalid JSON. Using simple fallback.")
            scored_candidates = simple_ranking_fallback(request_data, candidates)
        else:
            scored_candidates = result.get("scored_candidates", [])
        if not scored_candidates:
            # Fallback check: sometimes models use different keys
            for key in ["candidates", "results", "scores"]:
                if key in result:
                    scored_candidates = result[key]
                    break
        
        best_match = None
        best_confidence = 0.0
        scored_list = []
        
        for sc in scored_candidates:
            pid = sc.get("plan_id")
            conf = float(sc.get("confidence_score", 0.0))
            
            # find original candidate
            matched_cand = next((c for c in candidates if str(c["plan_id"]) == str(pid)), None)
            if matched_cand:
                # Add confidence to candidate record
                cand_copy = matched_cand.copy()
                cand_copy["confidence_score"] = conf
                scored_list.append(cand_copy)
                
                # track the best match
                if conf > best_confidence:
                    best_confidence = conf
                    best_match = cand_copy

        # Sort the scored list by confidence descending
        scored_list.sort(key=lambda x: x["confidence_score"], reverse=True)

        if best_match:
            _best_id = best_match.get('plan_id', 'Unknown') if isinstance(best_match, dict) else 'Unknown'
            log.info(f"LLM Top Choice: ID={_best_id}, Confidence={best_confidence:.2f}")
        else:
            log.info("LLM Top Choice: None")

        return {
            "match": best_match,
            "confidence_score": best_confidence,
            "scored_candidates": scored_list
        }

    except Exception as exc:
        err_msg = str(exc)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            err_msg = "Gemini API Quota Exceeded. Please try switching to a different model (e.g., gemini-2.0-flash) in the sidebar or wait a few minutes."
        elif "503" in err_msg or "UNAVAILABLE" in err_msg:
            err_msg = "Gemini API is currently overloaded. Please try again in a few moments or switch models."
        log.error(f"LLM final match ranking failed: {err_msg}")
        return {"match": None, "confidence_score": 0.0, "reasoning": f"Error: {err_msg}"}


# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────
def query_plan_optimizer(payer_info: List[str], plan_name: str,
                         plan_sub_type: Optional[str] = None,
                         group_name:    Optional[str] = None,
                         state_name:    Optional[str] = None) -> Dict:
    """
    Coordinates the full matching pipeline: Exact Match -> LLM Discovery -> LLM Ranking.

    Args:
        payer_info (List[str]): List of possible payer names.
        plan_name (str): The plan name from eligibility data.
        plan_sub_type (Optional[str]): PPO, HMO, etc.
        group_name (Optional[str]): Group identifier if available.
        state_name (Optional[str]): Geographic state filter.

    Returns:
        Dict: Final match result with metadata and confidence scores.
    """
    # ── Deterministic Partner Injection ──────────────────
    extended_payers = list(payer_info)
    p_lower = [p.lower() for p in extended_payers]
    
    # Cigna <-> HealthSpring
    if any("cigna" in p or "healthspring" in p for p in p_lower):
        if "cigna" not in "".join(p_lower): extended_payers.append("Cigna")
        if "healthspring" not in "".join(p_lower): extended_payers.append("HealthSpring")
        
    # Aetna <-> Coventry <-> Meritain
    if any("aetna" in p or "coventry" in p or "meritain" in p for p in p_lower):
        if "aetna" not in "".join(p_lower): extended_payers.append("Aetna")
        if "coventry" not in "".join(p_lower): extended_payers.append("Coventry")
        if "meritain" not in "".join(p_lower): extended_payers.append("Meritain Health")

    # Highmark <-> BCBS PA
    if any("highmark" in p or "bcbs" in p or "blue cross" in p for p in p_lower):
        if "highmark" not in "".join(p_lower): extended_payers.append("Highmark")
        if "bcbs" not in "".join(p_lower): extended_payers.append("BCBS Pennsylvania")

    extended_payers = list(dict.fromkeys(extended_payers))

    request_data = {
        "payer_info":    extended_payers,
        "plan_name":     plan_name,
        "plan_sub_type": plan_sub_type,
        "group_name":    group_name,
        "state_name":    state_name,
    }
    log.debug(f"Starting optimization pipeline for: {request_data}")

    conn = None
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=RealDictCursor)

        # ── Phase 1: Exact Match ──────────────────────────────
        log.info("Phase 1 – Attempting exact database match...")
        match = find_exact_match(cur, payer_info, plan_name, plan_sub_type, state_name)
        if match:
            log.info(f"✔ Exact Match Found: plan_id={match['plan_id']}")
            # We no longer short-circuit here so that the full discovery and ranking
            # pipeline runs, ensuring the 'Ranked Candidates' table is always populated.
            # return {"match": match, "confidence_score": 1.0, "method": "exact_match"}

        def generate_fallback_sql(data: Dict) -> str:
            """Broad multi-payer search as a fallback if LLM fails."""
            p_vars = data.get('payer_info', [])
            state  = data.get('state_name', '')
            
            p_filters = " OR ".join([f"payer_name ILIKE '%{p}%'" for p in p_vars])
            # Add general partner hint
            if any("cigna" in str(x).lower() for x in p_vars):
                p_filters += " OR payer_name ILIKE '%HealthSpring%'"
            if any("aetna" in str(x).lower() for x in p_vars):
                p_filters += " OR payer_name ILIKE '%Coventry%' OR payer_name ILIKE '%Meritain%'"
            
            # BROAD: Search by Payer + State only (no plan name filter)
            # Phase 3 (AI Ranking) handles plan-level matching.
            sql = f"SELECT * FROM ebv_genai.plan_details WHERE ({p_filters})"
            if state:
                s_vars = get_state_variants(state)
                s_filters = " OR ".join([f"state_name ILIKE '%{v}%'" for v in s_vars])
                sql += f" AND ({s_filters})"
            sql += " LIMIT 100"
            return sql

        def run_search_and_rank(data: Dict) -> Dict:
            # Generate and Validate SQL
            sql = generate_discovery_sql(data)
            if not sql:
                log.warning("Discovery SQL generation returned None. Using fallback.")
                sql = generate_fallback_sql(data)
            
            if not validate_sql_safety(sql):
                log.warning(f"Unsafe SQL blocked: {sql}")
                return {"match": None, "confidence_score": 0.0, "candidates": [], "reasoning": "SQL safety validation failed."}

            log.info(f"Executing Discovery SQL: {sql}")
            
            # Execute and Fetch
            cur.execute(sql)
            cands = [dict(r) for r in cur.fetchall()]
            log.info(f"Discovery SQL returned {len(cands)} candidate(s).")
            
            # ── Automatic Broadening ──
            # If we got too few candidates, re-run with fallback (payer+state only)
            if len(cands) < 5:
                log.info(f"Too few candidates ({len(cands)}). Broadening search with fallback SQL...")
                fallback_sql = generate_fallback_sql(data)
                if fallback_sql != sql and validate_sql_safety(fallback_sql):
                    log.info(f"Executing Broadened SQL: {fallback_sql}")
                    cur.execute(fallback_sql)
                    broad_cands = [dict(r) for r in cur.fetchall()]
                    log.info(f"Broadened SQL returned {len(broad_cands)} candidate(s).")
                    # Merge: keep originals + add new unique ones
                    existing_ids = {str(c['plan_id']) for c in cands}
                    for bc in broad_cands:
                        if str(bc['plan_id']) not in existing_ids:
                            cands.append(bc)
                            existing_ids.add(str(bc['plan_id']))

            # Limit candidates to the top 20 most relevant to prevent AI confusion/hallucination
            essential_cols = ['plan_id', 'plan_name', 'payer_name', 'state_name', 'Updated_s3_frozen_pdf_url']
            reduced_cands = []
            for c in cands[:20]:
                reduced_cands.append({k: v for k, v in c.items() if k in essential_cols})

            # LLM Ranking
            rank_res = get_llm_final_match(data, reduced_cands)
            rank_res["candidates"] = cands # Keep the full list for raw display
            return rank_res

        # ── Phase 2 & 3: Primary Match (Payer + State) ─────────
        log.info(f"Phase 2 – Primary search (Payer + State: {state_name or 'Any'})...")
        res1 = run_search_and_rank(request_data)
        
        best_match  = res1["match"]
        confidence  = res1["confidence_score"]
        candidates  = res1["candidates"]
        method_label = "llm_discovery_and_match"

        # ── Final Results Mapping ──────────────────────────────
        # ── Final Results Mapping & Merging ────────────────────────
        # 1. Initialize scored_candidates with ALL discovery candidates (at 0% score)
        scored_candidates_map = {}
        for c in candidates:
            c_copy = dict(c)
            # Default to very low score if AI didn't rank it
            c_copy["confidence_score"] = 0.0
            scored_candidates_map[str(c_copy["plan_id"])] = c_copy
            
        # 2. Update with AI rankings from Phase 3
        for sc in res1.get("scored_candidates", []):
            pid = str(sc["plan_id"])
            if pid in scored_candidates_map:
                scored_candidates_map[pid]["confidence_score"] = sc["confidence_score"]
                
        # 3. Force Phase 1 Exact Match to 1.0 if it exists
        if match:
            pid = str(match["plan_id"])
            if pid not in scored_candidates_map:
                m_copy = dict(match)
                m_copy["confidence_score"] = 1.0
                scored_candidates_map[pid] = m_copy
            else:
                scored_candidates_map[pid]["confidence_score"] = 1.0
                
        # 4. Final list and top match selection
        final_scored_candidates = list(scored_candidates_map.values())
        # Sort by confidence
        final_scored_candidates.sort(key=lambda x: x.get("confidence_score", 0.0), reverse=True)
        
        if final_scored_candidates:
            top_rec = final_scored_candidates[0]
            best_f_match = top_rec
            f_confidence = top_rec["confidence_score"]
        else:
            best_f_match = None
            f_confidence = 0.0
            
        return {
            "match": best_f_match,
            "confidence_score": f_confidence,
            "method": method_label,
            "scored_candidates": final_scored_candidates,
            "all_db_records": candidates,
            "candidate_count": len(candidates),
            "payer_info": payer_info # Inject for UI Category reference
        }

    except Exception as exc:
        log.error(f"Pipeline execution error: {exc}")
        return {"error": str(exc), "match": None, "confidence_score": 0.0, "method": "error"}

    finally:
        if conn:
            conn.close()


# ─────────────────────────────────────────────────────────────
# BULK PROCESSOR
# ─────────────────────────────────────────────────────────────
def find_possible_matches():
    """
    Main entry point for bulk processing insurance plans from an Excel file.
    Iterates through 'Processed' rows and writes results to a new workbook.
    """
    log.info("=" * 70)
    log.info("PLAN OPTIMIZER – BULK RUN INITIATED")
    log.info(f"Input: {INPUT_FILE} | Output: {OUTPUT_FILE}")
    log.info("=" * 70)

    if not os.path.exists(INPUT_FILE):
        log.error(f"Input file missing: {INPUT_FILE}")
        return

    log.info("Reading Excel workbook...")
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    
    # Processed rows
    processed_mask = df["Eligibility Status"] == "Processed"
    processed_df = df[processed_mask]
    
    # Inactive/Skipped rows
    skipped_df = df[~processed_mask]

    results = []
    counts = {"match": 0, "none": 0, "error": 0, "skipped": 0}

    # First, handle skipped rows
    for index, row in skipped_df.iterrows():
        excel_row = index + 2
        results.append({
            "Excel Row": excel_row,
            "Original Plan": str(row.get("Plan Name (Eligibility)", "")),
            "Original Payer": str(row.get("Payer Name", "")),
            "Matched Plan": "SKIPPED",
            "Matched Payer": "SKIPPED",
            "Plan ID": "N/A",
            "Confidence": 0.0,
            "Method": "skipped_inactive",
            "Reasoning": f"Row status is '{row.get('Eligibility Status')}', not 'Processed'."
        })
        counts["skipped"] += 1

    # Then, process active rows
    if not processed_df.empty:
        for index, row in processed_df.iterrows():
            excel_row = index + 2
            log.info("-" * 60)
            log.info(f"Processing Excel Row {excel_row}")

            payers = list(dict.fromkeys(p.strip() for p in [str(row.get("Payer Name", "")), str(row.get("Resolved Payer Name", ""))] if p.strip()))
            plan_name = str(row.get("Plan Name (Eligibility)", "")).strip()
            subtype = infer_subtype(str(row.get("Policy Type", "")), plan_name)

            try:
                res = query_plan_optimizer(
                    payer_info=payers,
                    plan_name=plan_name,
                    plan_sub_type=subtype,
                    group_name=str(row.get("Group Name", "")).strip() or None,
                )

                if "error" in res:
                    raise RuntimeError(res["error"])

                match = res.get("match")
                confidence = res.get("confidence_score", 0.0)
                
                if match:
                    counts["match"] += 1
                else:
                    counts["none"] += 1

                results.append({
                    "Excel Row": excel_row,
                    "Original Plan": plan_name,
                    "Original Payer": " | ".join(payers),
                    "Matched Plan": match.get("plan_name") if match else "NO MATCH",
                    "Matched Payer": match.get("payer_name") if match else "NO MATCH",
                    "Plan ID": match.get("plan_id") if match else "N/A",
                    "Confidence": round(confidence, 4),
                    "Method": res.get("method"),
                    "Reasoning": res.get("reasoning", "N/A"),
                })

            except Exception as exc:
                counts["error"] += 1
                log.error(f"Error on row {excel_row}: {exc}")
                results.append({
                    "Excel Row": excel_row,
                    "Original Plan": plan_name,
                    "Matched Plan": "ERROR",
                    "Method": "error",
                    "Reasoning": str(exc),
                })

    if results:
        # Sort by Excel Row to maintain order
        final_df = pd.DataFrame(results).sort_values("Excel Row")
        final_df.to_excel(OUTPUT_FILE, index=False)
        log.info(f"Results exported to {OUTPUT_FILE}")

    log.info("=" * 70)
    log.info(f"FINAL SUMMARY: Matches={counts['match']}, No Match={counts['none']}, Errors={counts['error']}, Skipped={counts['skipped']}")
    log.info("=" * 70)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    find_possible_matches()