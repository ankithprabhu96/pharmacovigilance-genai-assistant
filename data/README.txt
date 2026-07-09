Agentic AI for Pharmacovigilance Case Triage
============================================

Category: Agentic AI

Data Files: case_medical_review_notes.csv, medical_knowledge_base.json, medwatch_regulatory_reports.csv, patients.csv, pharmacovigilance_cases.csv, prescriptions.csv, providers.csv


PROBLEM STATEMENT
-----------------

Drug safety teams receive adverse event reports around the clock. Triage agents assess seriousness hints, assign priority, retrieve relevant medical knowledge, and route cases to qualified reviewers before regulatory clocks expire.
Agentic triage goes beyond classification by executing workflow steps: duplicate detection, narrative summarization, MedWatch preparation flags, and provider notifications.
Students use pharmacovigilance_cases.csv, medical_knowledge_base.json, prescriptions, review notes, regulatory reports, patients, and providers to simulate end-to-end triage.


OBJECTIVES
----------

1. Design a LangGraph state machine with agent nodes for case intake, seriousness assessment, and routing; ingest pharmacovigilance_cases.csv into shared state with patient context from patients.csv.
2. Implement LangGraph tool nodes for medical_knowledge_base.json retrieval by drug and event, and narrative summarization from case_medical_review_notes.csv for reviewer briefings.
3. Configure agent nodes that assign priority using serious, severity, and priority_score fields with override rules and route to providers.csv specialties when clinician follow-up is required.
4. Orchestrate end-to-end PV triage workflows from case trigger through knowledge retrieval, priority assignment, MedWatch deadline flagging via medwatch_regulatory_reports.csv, and provider routing.
5. Configure human-in-the-loop escalation nodes for serious cases requiring medical reviewer sign-off; evaluate triage accuracy, time-to-first-action proxy, and reviewer load balance with trace observability.
6. Deploy the Agentic AI PV triage agent with documented LangGraph state graph, tool schemas, and tracing configuration.


NOTES
-----

Technology Reference
~~~~~~~~~~~~~~~~~~
  - LangGraph state machine with PV triage agent nodes, clinical knowledge-retrieval tools, priority-routing conditional edges, and human-in-the-loop reviewer checkpoints; end-to-end workflow from adverse event trigger through triage and routing with observability/tracing.

Dataset Descriptions
~~~~~~~~~~~~~~~~~~~~
  - case_medical_review_notes.csv: Medical reviewer notes and assessment text linked to pharmacovigilance case IDs.
  - medical_knowledge_base.json: Approved clinical and patient education articles for RAG retrieval with topic tags.
  - medwatch_regulatory_reports.csv: Regulatory reporting metadata: submission deadlines, report status, and case linkages.
  - patients.csv: Patient demographics, comorbidities, BMI, and baseline clinical attributes.
  - pharmacovigilance_cases.csv: Adverse event case reports with drug, event narrative, severity, seriousness flag, and priority score.
  - prescriptions.csv: Medication orders linked to patients and providers with drug codes and fill dates.
  - providers.csv: Healthcare provider directory with specialty, department, capacity, and location attributes.

How Datasets Relate and Join
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Link pharmacovigilance_cases.csv to clinical tables via patient_id and case_id fields.

Suggested ML / Analytics Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Design a tool-using agent with explicit planning: parse user intent, select tools (SQL/CSV queries, policy retrieval, record updates), execute steps, and verify outputs against schema. Ground all policy and medical content in JSON/CSV sources. Log traces for audit.

Evaluation Metrics
~~~~~~~~~~~~~~~~~~
Task completion rate, grounded answer rate, citation accuracy, escalation rate, and SLA compliance.

Student Deliverables
~~~~~~~~~~~~~~~~~~~~
- Data exploration notebook or report documenting schema, missing values, and join diagrams for all project files.
- Implemented pipeline (Python preferred) reproducible from raw CSV/JSON/JSONL/DB files in this folder.
- Model, agent, or analytics outputs with held-out evaluation using the metrics above.
- Written summary (2-3 pages) interpreting results, limitations, and recommended production next steps.
- Artifact export appropriate to project type: scored CSV, recommendation lists, agent trace logs, dashboard screenshots, or generated report samples.

Technical Notes
~~~~~~~~~~~~~~~
- All data is synthetic and intended for education, portfolio demonstrations, and prototyping.
- Do not assume external APIs or live systems; simulate tool calls against local files.
- When using GenAI components, document prompts, retrieval configuration, and safety guardrails explicitly.
- Preserve reproducibility: set random seeds, document train/validation splits, and version any embedding models used.
