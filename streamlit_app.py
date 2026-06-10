import streamlit as st
from llm import get_answer

st.set_page_config(
    page_title="Clinical Trial Navigator",
    page_icon="🏥",
    layout="wide"
)

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

tab1, tab2 = st.tabs(["🔬 Research Chat", "⚠️ Adverse Events Analysis"])

with tab1:
    st.title("Clinical Trial Research Navigator")
    st.caption("Ask questions about clinical trials, drugs, side effects, and eligibility")

    with st.sidebar:
        st.header("Search Filters")
        condition_filter = st.text_input("Condition (optional)", placeholder="e.g. diabetes")
        drug_filter = st.text_input("Drug Name (optional)", placeholder="e.g. metformin")
        phase_options = ["All", "Phase 1", "Phase 2", "Phase 3", "Phase 4"]
        phase_filter = st.selectbox("Phase", phase_options)
        top_k = st.slider("Number of Results", min_value=3, max_value=10, value=5)

        if st.button("Clear Chat"):
            st.session_state.messages = []
            from llm import store
            store.pop(st.session_state.session_id, None)
            st.rerun()

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

                if message.get("has_contradiction"):
                    st.warning("⚠️ Contradictions detected across trials")
                    for c in message.get("contradictions", []):
                        severity_color = {
                            "Critical":    "🔴",
                            "Significant": "🟡",
                            "Minor":       "🟢",
                        }.get(c["severity"], "⚪")

                        with st.expander(f"{severity_color} {c['severity']} contradiction between {c['trial_1']} and {c['trial_2']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**{c['trial_1']} claims:**")
                                st.write(c["claim_1"])
                            with col2:
                                st.markdown(f"**{c['trial_2']} claims:**")
                                st.write(c["claim_2"])

                if message.get("chunks"):
                    with st.expander("View Sources"):
                        for chunk in message["chunks"]:
                            st.markdown(
                                f"**Trial ID:** {chunk['doc_id']} | "
                                f"**Section:** {chunk['section']} | "
                                f"**RRF Score:** {chunk.get('rrf_score', 0):.4f}"
                            )
                            st.caption(chunk["title"])

    question = st.chat_input("Ask about clinical trials...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Searching trials and generating answer..."):
            result = get_answer(
                query=question,
                session_id=st.session_state.session_id,
                condition=condition_filter if condition_filter else None,
                drug_name=drug_filter if drug_filter else None,
                phase=phase_filter if phase_filter != "All" else None,
                top_k=top_k,
            )

        st.session_state.messages.append({
            "role":              "assistant",
            "content":           result["answer"],
            "chunks":            result["chunks"],
            "has_contradiction": result["has_contradiction"],
            "contradictions":    result["contradictions"],
        })
        st.rerun()

with tab2:
    st.title("Adverse Events Analysis")
    st.caption("Search for drug side effects across clinical trials")

    ae_drug = st.text_input("Drug Name", placeholder="e.g. metformin", key="ae_drug")
    ae_condition = st.text_input("Condition (optional)", placeholder="e.g. diabetes", key="ae_condition")

    if st.button("Analyze Adverse Events"):
        if not ae_drug:
            st.warning("Please enter a drug name.")
        else:
            with st.spinner("Analyzing adverse events..."):
                result = get_answer(
                    query=f"adverse events and side effects of {ae_drug}",
                    session_id="ae_analysis_tab",
                    condition=ae_condition if ae_condition else None,
                    drug_name=ae_drug,
                    top_k=10,
                )

            ae_chunks = [c for c in result["chunks"] if c.get("section") == "AdverseEvents"]

            if not result["chunks"]:
                st.warning("No adverse event data found for this drug.")
            else:
                st.markdown(result["answer"])

                if ae_chunks:
                    with st.expander("Raw Adverse Event Data"):
                        for chunk in ae_chunks:
                            st.markdown(f"**{chunk['title']}** (`{chunk['doc_id']}`)")
                            st.code(chunk["content"])
