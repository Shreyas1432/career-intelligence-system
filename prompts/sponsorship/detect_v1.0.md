---
version: "1.0.0"
description: "AI-fallback prompt template to analyze visa sponsorship signals from job descriptions"
input_variables:
  - job_description
---
# Visa Sponsorship Analysis

Analyze the following job description to identify visa sponsorship status, work authorization requirements, and related relocation or international team signals.

## Job Description
```text
{{ job_description }}
```

## Task Instructions
1. Search the text for any clues related to:
   - **sponsorship_mention**: Direct statements about visa sponsorship (e.g. "We provide visa sponsorship", "H1B sponsorship available", "We cannot sponsor visas").
   - **work_auth**: Requirements for US citizenship, green card, right to work, or legal authorization to work in the country (e.g. "must be authorized to work", "US citizen or Green Card holder only").
   - **international_workforce**: Mentions of international employees, cross-border hiring, or sponsoring foreign nationals.
   - **relocation**: Explicit relocation support or relocation packages.
   - **global_team**: References to global, multinational, or highly distributed international teams.

2. Classify the overall sponsorship status as one of:
   - `positive`: Explicitly states that visa sponsorship is available/offered.
   - `negative`: Explicitly states that visa sponsorship is NOT available/offered, or requires US citizenship/permanent residency only.
   - `neutral`: Contains standard work authorization requirements or indirect clues (like relocation/global team) but does not explicitly state if sponsorship is offered or denied.
   - `unknown`: No sponsorship, work auth, or international signals of any kind are present.

3. Extract exact snippets matching each signal, assign confidence scores (0.0 to 1.0), and specify if the signal is positive (supportive of hiring foreign nationals) or negative (restrictive).

4. Ensure your response matches the requested JSON schema.
