# Phase 1: Data Ingestion Foundation - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers the data ingestion pipeline. It allows reading historical CSV/Excel datasets, parsing dates, handling missing values, and outputting clean Pandas DataFrames ready for ML modeling.
(New capabilities like API integration belong in other phases.)

</domain>

<decisions>
## Implementation Decisions

### Ingestion Interface
- Use a local Python script/CLI first for fastest iteration before building REST endpoints.
- Use Pandas as the primary Data Manipulation engine due to ecosystem standard.

### Data Normalization
- Handle missing values using forward-fill for time-series economic data, with explicit console warnings.
- Standardize all time-series to daily frequency.

### Claude's Discretion
- Code architecture and specific directory layout for scripts (e.g. `src/data/ingest.py`).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Scope
- `.planning/PROJECT.md` — Core MVP scope restrictions.
- `.planning/REQUIREMENTS.md` — Data requirements (DATA-01, DATA-02).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None (Empty codebase).

### Established Patterns
- Python standard library with basic requirements.txt.

### Integration Points
- This lays the foundation. Will be imported by Phase 2 ML models.

</code_context>

<specifics>
## Specific Ideas
- The user specified "upload static CSVs/Excel files for historical and backtested data (fastest to build MVP)". Ensure pandas `read_csv` and `read_excel` are supported.
</specifics>

<deferred>
## Deferred Ideas
None — discussion stayed within phase scope
</deferred>

---

*Phase: 1-data-ingestion-foundation*
*Context gathered: 2026-03-19*
