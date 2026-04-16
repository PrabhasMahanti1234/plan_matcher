import os
import sys
import time
import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


if __name__ == "__main__" and os.environ.get("PLAN_MATCHER_STREAMLIT_BOOTSTRAP") != "1":
    os.environ["PLAN_MATCHER_STREAMLIT_BOOTSTRAP"] = "1"
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    raise SystemExit(stcli.main())

# Import the core matching logic
try:
    from llm_plan_matcher import query_plan_optimizer, setup_logging, infer_subtype, get_db_connection, get_state_variants
    from psycopg2.extras import RealDictCursor
except ImportError:
    st.error("Error: Could not import core logic from `llm_plan_matcher.py`. Ensure the file is in the same directory.")
    st.stop()

def get_all_payer_plans(payer_name, state_name=None):
    """Retrieves all available plans for a given payer and state from the database."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Clean up seeker terms
            search_payer = payer_name.strip()
            
            query = 'SELECT DISTINCT plan_id, plan_name, plan_type, plan_sub_type, state_name, "Updated_s3_frozen_pdf_url" FROM ebv_genai.plan_details WHERE payer_name ILIKE %s'
            params = [f"%{search_payer}%"]
            
            if state_name and state_name.strip():
                query += " AND state_name ILIKE %s"
                params.append(f"%{state_name.strip()}%")
            
            query += " ORDER BY plan_name ASC LIMIT 200"
            cur.execute(query, params)
            return cur.fetchall()
    finally:
        conn.close()

# Load env variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Insurance Plan Matcher", layout="wide")

st.title("🛡️ Insurance Plan Matcher")
st.markdown("""
Refinement and identification of insurance plans using AI-driven matching and a comprehensive plan database.
""")

# Sidebar Configuration
st.sidebar.header("Configuration")
st.sidebar.info("Model: **Llama 3 (8B)**")
st.sidebar.markdown("Powered by **AWS Bedrock**")

CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Create Tabs
tab_matcher, tab_explorer = st.tabs(["🔍 Plan Matcher", "🗄️ Plan Database Explorer"])

with tab_matcher:
    # Existing Matcher UI
    st.subheader("🔍 Single Plan Lookup")
    st.markdown("Enter details manually to find the best database match using LLM logic.")

    with st.form("manual_entry_form"):
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            m_payer = st.text_input("Payer Name", placeholder="e.g., Aetna, BCBS")
            m_plan = st.text_input("Plan Name", placeholder="e.g., Choice POS II")
        with m_col2:
            m_subtype = st.text_input("Subtype (Optional)", placeholder="e.g., PPO, HMO")
            m_group = st.text_input("Group Name/Number (Optional)")
        with m_col3:
            m_state = st.text_input("State Name (Optional)", placeholder="e.g., Pennsylvania, PA")
        
        submit_btn = st.form_submit_button("Search for Match")
        
    if submit_btn:
        if not m_payer or not m_plan:
            st.error("Please enter both Payer Name and Plan Name.")
        else:
            with st.status("🧠 Analyzing the best possible match...", expanded=True) as status:
                st.write("Running AI matching optimization...")
                try:
                    res = query_plan_optimizer(
                        payer_info=[m_payer],
                        plan_name=m_plan,
                        plan_sub_type=m_subtype if m_subtype else None,
                        group_name=m_group if m_group else None,
                        state_name=m_state if m_state else None
                    )
                    match = res.get("match")
                    confidence = res.get("confidence_score", 0.0)
                    method = res.get("method", "unknown")
                    status.update(label="AI Matching Complete!", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Error during LLM matching: {e}")
                    match = None
                    res = None

            st.divider()
            if match:
                match_plan_name = match.get('plan_name', 'N/A')
                match_plan_id = match.get('plan_id', 'N/A')
                
                if method == "llm_discovery_and_match" or method == "exact_match":
                    st.success(f"### 🎯 The closest match: {match_plan_name} (ID: {match_plan_id})")
                else:
                    st.warning(f"### ⚠️ {method}: {match_plan_name} (ID: {match_plan_id})")
                
                m_res_col1, m_res_col2 = st.columns(2)
                with m_res_col1:
                    st.write("**Record Details:**")
                    st.write(f"- **Official Plan Name:** `{match_plan_name}`")
                    st.write(f"- **Database Plan ID:** `{match_plan_id}`")
                    st.write(f"- **Payer:** {match.get('payer_name', 'N/A')}")
                    st.write(f"- **Region:** {match.get('state_name', 'N/A')}")
                with m_res_col2:
                    st.write("**Intelligence Context:**")
                    st.write(f"- **Confidence Score:** `{confidence:.2%}`")
                    st.write(f"- **Match Status:** `{method}`")
                    st.write(f"- **Candidates Scored:** {res.get('candidate_count', 1) if res else 'N/A'}")
            else:
                st.error("### ❌ no_candidates")
                st.warning("The LLM could not identify a single definitive match for this search.")
            
            scored_candidates = res.get("scored_candidates", []) if res else []
            if scored_candidates:
                st.subheader(f"📊 Top Ranked Candidates ({len(scored_candidates)} matched)")
                sc_df = pd.DataFrame(scored_candidates)
                
                def get_category(name, input_payers):
                    n_p = str(name).lower().strip()
                    input_ps = [str(x).lower().strip() for x in input_payers]
                    if any(ip in n_p or n_p in ip for ip in input_ps):
                        return "🏢 Direct"
                    has_input_cigna = any("cigna" in ip or "healthspring" in ip for ip in input_ps)
                    if has_input_cigna and ("cigna" in n_p or "healthspring" in n_p):
                        return "🤝 Alias"
                    has_input_highmark = any("highmark" in ip or "blue cross" in ip or "bcbs" in ip for ip in input_ps)
                    if has_input_highmark and ("highmark" in n_p or "blue cross" in n_p or "bcbs" in n_p):
                        return "🤝 Alias"
                    has_input_aetna = any("aetna" in ip for ip in input_ps)
                    if has_input_aetna and ("aetna" in n_p or "coventry" in n_p or "meritain" in n_p):
                        return "🤝 Alias"
                    return "🔗 Partner"
                
                sc_df['Category'] = sc_df['payer_name'].apply(lambda x: get_category(x, res.get('payer_info', [])))
                sc_df['confidence_score'] = sc_df['confidence_score'].apply(lambda x: f"{float(x):.2%}")
                
                cols = ['Category', 'payer_name', 'plan_id', 'plan_name', 'state_name', 'confidence_score']
                if 'Updated_s3_frozen_pdf_url' in sc_df.columns:
                    cols.append('Updated_s3_frozen_pdf_url')
                
                display_sc_df = sc_df[cols].rename(columns={
                    "confidence_score": "Confidence Score", 
                    "payer_name": "Payer Name", 
                    "plan_id": "Plan ID", 
                    "plan_name": "Plan Name", 
                    "state_name": "State Name",
                    "Updated_s3_frozen_pdf_url": "Benefit Document"
                })
                
                st.dataframe(
                    display_sc_df, 
                    width="stretch", 
                    hide_index=True,
                    column_config={
                        "Benefit Document": st.column_config.LinkColumn("Benefit Document")
                    }
                )
            else:
                st.info("No AI ranked candidates found for this search.")
            
            st.divider()
            try:
                db_all = get_all_payer_plans(m_payer, m_state)
            except:
                db_all = []
            
            st.subheader(f"🗄️ Total Database Records found for '{m_payer}'")
            if db_all:
                st.write(f"📊 **Direct Database Found:** `{len(db_all)}` records in DB for this search.")
                candidates_df = pd.DataFrame(db_all)
                cols_to_show = ['plan_id', 'plan_name', 'state_name']
                if 'Updated_s3_frozen_pdf_url' in candidates_df.columns:
                    cols_to_show.append('Updated_s3_frozen_pdf_url')
                    
                display_df = candidates_df[cols_to_show].rename(columns={
                    "plan_id": "Plan ID",
                    "plan_name": "Plan Name",
                    "state_name": "State Name",
                    "Updated_s3_frozen_pdf_url": "Benefit Document"
                })
                st.dataframe(
                    display_df, 
                    width="stretch", 
                    hide_index=True,
                    column_config={
                        "Benefit Document": st.column_config.LinkColumn("Benefit Document")
                    }
                )
            else:
                st.info(f"No exact records found for '{m_payer}' and State: '{m_state or 'Any'}'.")

with tab_explorer:
    st.subheader("🗄️ Database Explorer")
    st.markdown("Browse and filter the complete `plan_details` database.")
    
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        e_payer = st.text_input("Filter by Payer", key="e_payer")
    with exp_col2:
        e_plan = st.text_input("Filter by Plan Name", key="e_plan")
    with exp_col3:
        e_state = st.text_input("Filter by State", key="e_state")
    
    @st.cache_data(ttl=600)
    def fetch_explorer_data(p=None, pl=None, s=None):
        conn = get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = 'SELECT plan_id, payer_name, plan_name, plan_type, state_name, "Updated_s3_frozen_pdf_url" FROM ebv_genai.plan_details'
                conditions = []
                params = []
                if p:
                    conditions.append("payer_name ILIKE %s")
                    params.append(f"%{p}%")
                if pl:
                    conditions.append("plan_name ILIKE %s")
                    params.append(f"%{pl}%")
                if s:
                    s_vars = get_state_variants(s)
                    s_conds = " OR ".join([f"state_name ILIKE %s" for _ in s_vars])
                    conditions.append(f"({s_conds})")
                    params.extend([f"%{v}%" for v in s_vars])
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY payer_name ASC, plan_name ASC LIMIT 500"
                cur.execute(query, params)
                return cur.fetchall()
        finally:
            conn.close()
            
    db_records = fetch_explorer_data(e_payer, e_plan, e_state)
    if db_records:
        st.write(f"Showing top {len(db_records)} matching records.")
        edf = pd.DataFrame(db_records)
        edf_display = edf.rename(columns={
            "plan_id": "Plan ID",
            "payer_name": "Payer Name",
            "plan_name": "Plan Name",
            "state_name": "State Name",
            "Updated_s3_frozen_pdf_url": "Benefit Document"
        })
        st.dataframe(
            edf_display,
            width="stretch",
            hide_index=True,
            column_config={
                "Benefit Document": st.column_config.LinkColumn("Benefit Document")
            }
        )
    else:
        st.info("No records match your filters.")
