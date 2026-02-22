# Optimization Notes: llm_prompt

**Rationale**

Adds a clear, stepwise rubric (parse → grade → compare), explicit tie/uncertainty handling to return Neither, emphasizes using only provided attributes, includes key attribute priorities and synonyms, and enforces strict JSON-only output.

**Changelog**

- Reorganized into inputs, decision steps, guidance, and output for clarity.
- Introduced Explicit/Partial/Irrelevant grading to standardize evidence assessment.
- Added tie-breaking rules and default to Neither when equal or uncertain.
- Specified key attribute priorities (type, dimensions/bed size, features, brand/model, set/quantity, material/color).
- Included common furniture synonyms and category distinctions to reduce false matches.
- Added guards against inference: use only name/description; ignore noise/marketing.
- Enforced strict JSON-only output with fixed keys and allowed values.
