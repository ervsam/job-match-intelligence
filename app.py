import streamlit as st
import json
from src.vectorstore import query
from src.reason import analyze_job, load_profile

import os

# Auto-ingest into ChromaDB if collection is empty (handles cold cloud start)
from src.vectorstore import ingest
ingest()

st.set_page_config(page_title="Job Match Intelligence", layout="wide")
st.title("🎯 Job Match Intelligence")
st.caption("RAG-powered job ranking and fit analysis from your Gmail inbox")

# --- Load data ---
@st.cache_data
def load_data():
    with open("data/profile_embedding.json") as f:
        profile_data = json.load(f)

    parsed_jobs = {}
    with open("data/parsed_jobs.jsonl") as f:
        for line in f:
            job = json.loads(line)
            parsed_jobs[job["subject"]] = job

    profile_text = load_profile()
    return profile_data, parsed_jobs, profile_text


@st.cache_data
def load_scored_jobs():
    jobs = []
    try:
        with open("data/scored_jobs.jsonl") as f:
            for line in f:
                jobs.append(json.loads(line))
    except FileNotFoundError:
        pass
    return jobs


profile_data, parsed_jobs, profile_text = load_data()
scored_jobs = load_scored_jobs()

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    n_results = st.slider("Number of jobs to show", 5, 30, 10)
    show_profile = st.checkbox("Show my profile")

    if show_profile:
        st.markdown("---")
        st.markdown(profile_text)

# --- Main: ranked job list ---
st.subheader(f"Top {n_results} matched jobs")

top_jobs = query(profile_data["embedding"], n_results=n_results)

# Merge with scored_jobs if available
scored_by_subject = {j["subject"]: j for j in scored_jobs}

for i, hit in enumerate(top_jobs, 1):
    subject = hit["subject"]
    company = hit["company"] or "Unknown"
    score = hit["score"]

    with st.expander(f"{i}. [{score:.3f}] {company} — {subject[:70]}"):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**Company:** {company}")
            st.markdown(f"**Role:** {subject}")
            st.markdown(f"**Date:** {hit['date'][:16]}")
            st.markdown(f"**Similarity score:** `{score:.4f}`")

        with col2:
            if subject in scored_by_subject:
                analysis = scored_by_subject[subject]["analysis"]
                fit = analysis.get("fit_score", "?")
                color = "🟢" if fit >= 8 else "🟡" if fit >= 6 else "🔴"
                st.markdown(f"**GPT-4o fit score:** {color} {fit}/10")

        if subject in scored_by_subject:
            analysis = scored_by_subject[subject]["analysis"]
            st.markdown("---")
            st.markdown("**Fit summary**")
            st.write(analysis.get("fit_summary", ""))

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Strengths**")
                for s in analysis.get("strengths", []):
                    st.markdown(f"- {s}")
            with col4:
                st.markdown("**Gaps**")
                gaps = analysis.get("gaps", [])
                if gaps:
                    for g in gaps:
                        st.markdown(f"- {g}")
                else:
                    st.markdown("None identified")

            st.markdown("**What to emphasize**")
            for e in analysis.get("emphasize", []):
                st.markdown(f"- {e}")

        elif subject in parsed_jobs:
            st.markdown("---")
            st.markdown("**Job body preview**")
            st.text(parsed_jobs[subject]["body"][:500] + "...")
            if st.button(f"Analyze with GPT-4o", key=f"analyze_{i}"):
                with st.spinner("Analyzing..."):
                    job = parsed_jobs[subject]
                    profile = load_profile()
                    analysis = analyze_job(profile, job)
                st.markdown("---")
                fit = analysis.get("fit_score", "?")
                color = "🟢" if fit >= 8 else "🟡" if fit >= 6 else "🔴"
                st.markdown(f"**GPT-4o fit score:** {color} {fit}/10")
                st.markdown("**Fit summary**")
                st.write(analysis.get("fit_summary", ""))
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**Strengths**")
                    for s in analysis.get("strengths", []):
                        st.markdown(f"- {s}")
                with col4:
                    st.markdown("**Gaps**")
                    gaps = analysis.get("gaps", [])
                    if gaps:
                        for g in gaps:
                            st.markdown(f"- {g}")
                    else:
                        st.markdown("None identified")
                st.markdown("**What to emphasize**")
                for e in analysis.get("emphasize", []):
                    st.markdown(f"- {e}")
