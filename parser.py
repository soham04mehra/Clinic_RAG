def _meta_header(trial_info: dict, section_name: str, brief_summary: str = "") -> str:
    """Build the standard metadata header with brief study context to prevent section fragmentation."""
    summary_snippet = f" | Focus: {brief_summary[:200]}" if brief_summary else ""
    return (
        f"Trial ID: {trial_info['nct_id']}\n"
        f"Title: {trial_info['title']}\n"
        f"Condition: {trial_info['condition']} | Drug: {trial_info['drug_name']} | "
        f"Phase: {trial_info['phase']} | Status: {trial_info['status']}{summary_snippet}\n"
        f"Section: {section_name}\n\n"
    )


def adverse_events_to_section(ae_data: dict, brief_summary: str = "") -> str:
    """Format the adverse events data dictionary into a readable text block with trial context."""
    summary_snippet = f" | Focus: {brief_summary[:200]}" if brief_summary else ""
    lines = [
        f"Trial ID: {ae_data['nct_id']}",
        f"Drug: {ae_data['drug']} | Condition: {ae_data['condition']} | Phase: {ae_data['phase']}{summary_snippet}",
        f"Total Patients: {ae_data['total_patients']} | Deaths Reported: {ae_data['deaths']}",
        "",
        "SERIOUS EVENTS:",
    ]

    if ae_data["serious_events"]:
        for ev in ae_data["serious_events"]:
            lines.append(
                f"- {ev['term']} | Organ: {ev['organ']} | "
                f"{ev['pct']}% ({ev['affected']}/{ev['at_risk']})"
            )
    else:
        lines.append("- No serious events reported")

    lines.append("")
    lines.append("OTHER EVENTS:")

    if ae_data["other_events"]:
        for ev in ae_data["other_events"]:
            lines.append(
                f"- {ev['term']} | Organ: {ev['organ']} | "
                f"{ev['pct']}% ({ev['affected']}/{ev['at_risk']})"
            )
    else:
        lines.append("- No other events reported")

    return "\n".join(lines)


def trial_to_sections(trial: dict, condition: str) -> list[dict]:
    """
    Convert a raw trial Study dict into 5 row-ready dicts (one per section).
    Returns an empty list if the trial cannot be parsed.
    """
    try:
        # API v2 uses lowercase camelCase keys
        protocol = trial.get("protocolSection") or trial.get("ProtocolSection") or {}
        results  = trial.get("resultsSection")  or trial.get("ResultsSection")  or {}

        id_module          = protocol.get("identificationModule")    or protocol.get("IdentificationModule")    or {}
        desc_module        = protocol.get("descriptionModule")       or protocol.get("DescriptionModule")       or {}
        design_module      = protocol.get("designModule")            or protocol.get("DesignModule")            or {}
        eligibility_module = protocol.get("eligibilityModule")       or protocol.get("EligibilityModule")       or {}
        outcomes_module    = protocol.get("outcomesModule")          or protocol.get("OutcomesModule")          or {}
        status_module      = protocol.get("statusModule")            or protocol.get("StatusModule")            or {}
        arms_module        = protocol.get("armsInterventionsModule") or protocol.get("ArmsInterventionsModule") or {}
        ae_module          = results.get("adverseEventsModule")      or results.get("AdverseEventsModule")      or {}

        nct_id = id_module.get("nctId")      or id_module.get("NCTId")      or "UNKNOWN"
        title  = id_module.get("briefTitle") or id_module.get("BriefTitle") or "No title available"

        # Find the Drug intervention name
        drug_name = "N/A"
        # API v2: arms_module["interventions"] is a list directly
        interventions = (
            arms_module.get("interventions")
            or arms_module.get("InterventionList", {}).get("Intervention", [])
        )
        for iv in interventions:
            iv_type = iv.get("type") or iv.get("InterventionType") or ""
            if iv_type.lower() == "drug":
                drug_name = iv.get("name") or iv.get("InterventionName") or "N/A"
                break

        # API v2: design_module["phases"] is a list directly
        phases = (
            design_module.get("phases")
            or design_module.get("PhaseList", {}).get("Phase", [])
        )
        phase = phases[0] if phases else "N/A"

        status = (
            status_module.get("overallStatus")
            or status_module.get("OverallStatus")
            or "N/A"
        )

        # Build outcomes text — API v2: outcomes_module["primaryOutcomes"] is a list
        primary_outcomes = (
            outcomes_module.get("primaryOutcomes")
            or outcomes_module.get("PrimaryOutcomeList", {}).get("PrimaryOutcome", [])
        )
        if primary_outcomes:
            outcomes_text = "\n".join(
                f"- {o.get('measure', o.get('PrimaryOutcomeMeasure', 'N/A'))} "
                f"(Time: {o.get('timeFrame', o.get('PrimaryOutcomeTimeFrame', 'N/A'))})"
                for o in primary_outcomes
            )
        else:
            outcomes_text = "No primary outcomes reported."

        trial_info = {
            "nct_id":    nct_id,
            "title":     title,
            "condition": condition,
            "drug_name": drug_name,
            "phase":     phase,
            "status":    status,
        }

        from fetcher import extract_adverse_events
        ae_data = extract_adverse_events(ae_module, trial_info)

        def make_row(chunk_index: int, section: str, content: str) -> dict:
            return {
                "doc_id":      nct_id,
                "doc_type":    "clinical_trial",
                "title":       title,
                "source_id":   nct_id,
                "condition":   condition,
                "drug_name":   drug_name,
                "phase":       phase,
                "status":      status,
                "section":     section,
                "chunk_index": chunk_index,
                "content":     content,
            }

        summary_text     = desc_module.get("briefSummary")       or desc_module.get("BriefSummary")           or "No summary available."
        detail_text      = desc_module.get("detailedDescription") or desc_module.get("DetailedDescription")    or "No detailed description available."
        eligibility_text = eligibility_module.get("eligibilityCriteria") or eligibility_module.get("EligibilityCriteria") or "No eligibility criteria available."

        rows = [
            make_row(0, "Summary",
                     _meta_header(trial_info, "Summary", summary_text) + summary_text),
            make_row(1, "Description",
                     _meta_header(trial_info, "Description", summary_text) + detail_text),
            make_row(2, "Eligibility",
                     _meta_header(trial_info, "Eligibility", summary_text) + eligibility_text),
            make_row(3, "Outcomes",
                     _meta_header(trial_info, "Outcomes", summary_text) + outcomes_text),
            make_row(4, "AdverseEvents",
                     adverse_events_to_section(ae_data, summary_text)),
        ]

        return rows

    except Exception as e:
        print(f"    ⚠ Skipping trial due to parse error: {e}")
        return []
