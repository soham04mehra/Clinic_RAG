from langchain_groq import ChatGroq
import os
import json
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0.0)


def group_chunks_by_trial(chunks: list[dict]) -> list[dict]:
    trials = {}
    for chunk in chunks:
        key = chunk["doc_id"]
        if key not in trials:
            trials[key] = {
                "doc_id":    chunk["doc_id"],
                "title":     chunk["title"],
                "condition": chunk["condition"],
                "drug_name": chunk["drug_name"],
                "phase":     chunk["phase"],
                "content":   [],
            }
        trials[key]["content"].append(chunk["content"])
    return list(trials.values())


def detect_contradictions(chunks: list[dict], question: str) -> dict:
    try:
        trials = group_chunks_by_trial(chunks)

        if len(trials) < 2:
            return {"has_contradiction": False, "contradictions": []}

        trials = trials[:4]

        trial_blocks = []
        for t in trials:
            combined_content = " ".join(t["content"])[:500]
            trial_blocks.append(
                f"Trial: {t['doc_id']}\n"
                f"Drug: {t['drug_name']}\n"
                f"Condition: {t['condition']}\n"
                f"Phase: {t['phase']}\n"
                f"Content: {combined_content}"
            )
        formatted_trials = "\n\n".join(trial_blocks)

        prompt = (
            "You are a medical research analyst.\n"
            "You are given excerpts from multiple clinical trials.\n"
            "Your job is to identify contradictions between trials.\n"
            "A contradiction is when two trials make conflicting claims about the same drug, condition, or outcome.\n\n"
            f"User question: {question}\n\n"
            f"Trial excerpts:\n{formatted_trials}\n\n"
            "Identify any contradictions and respond in this exact JSON format with no extra text:\n"
            "{\n"
            '    "has_contradiction": true or false,\n'
            '    "contradictions": [\n'
            "        {\n"
            '            "trial_1": "NCT ID of first trial",\n'
            '            "trial_2": "NCT ID of second trial",\n'
            '            "claim_1": "what trial 1 claims",\n'
            '            "claim_2": "what trial 2 claims",\n'
            '            "severity": "Minor" or "Significant" or "Critical"\n'
            "        }\n"
            "    ]\n"
            "}\n\n"
            "If no contradictions found return:\n"
            "{\n"
            '    "has_contradiction": false,\n'
            '    "contradictions": []\n'
            "}"
        )

        response = model.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw)

    except Exception as e:
        print(f"Contradiction detection error: {e}")
        return {"has_contradiction": False, "contradictions": []}
