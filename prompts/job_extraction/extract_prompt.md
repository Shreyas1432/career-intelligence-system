---
version: "1.0.0"
description: "Low-token job posting extraction prompt for local ScrapeGraphAI/Ollama runs"
input_variables:
  - source_url
  - max_skills
---
# Job Posting Extraction

Extract only facts explicitly supported by the supplied job posting source.

Source URL: {{ source_url }}

Return one JSON object matching the provided schema.

Rules:
- Do not infer or embellish missing details.
- Use `null` when a scalar field is not explicitly present.
- Use `[]` when list evidence is not explicitly present.
- Keep values concise and copy short phrases from the posting where useful.
- Prefer normalized skills, not full sentences.
- Limit `skills` to the {{ max_skills }} strongest explicit requirements.
- Use enum values exactly where possible.
- Set `confidence_score` from 0.0 to 1.0 based only on extraction certainty and visible evidence.

Extract:
- `company`: hiring company name.
- `title`: job title.
- `skills`: tools, technologies, methods, certifications, or core competencies.
- `experience_required`: explicit years or seniority requirement.
- `location`: work location, remote/hybrid/onsite region, or office.
- `visa_signal`: one of `sponsorship_available`, `no_sponsorship`, `work_auth_required`, `unknown`.
- `employment_type`: one of `full_time`, `part_time`, `contract`, `internship`, `temporary`, `freelance`, `unknown`.
- `domain`: one of `software_engineering`, `data_ai`, `product`, `design`, `sales`, `marketing`, `finance`, `operations`, `security`, `infrastructure`, `healthcare`, `education`, `legal`, `other`, `unknown`.
- `confidence_score`: number between 0.0 and 1.0.
