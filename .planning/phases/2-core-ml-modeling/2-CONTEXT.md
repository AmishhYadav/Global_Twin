# Phase 2: Core ML Modeling - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers the core machine learning models. It builds multi-variable dependency learning on top of Phase 1's cleaned datasets. It trains models that predict variables at T+1 using historical context, runs baseline evaluation metrics, and extracts feature importances to be used later in Phase 3 for the graph structure.

(New capabilities like adding graph queries belong in Phase 3.)
</domain>

<decisions>
## Implementation Decisions

### Model Architecture
- Train separate Random Forest models for each target variable, rather than a single multi-output model.

### Feature Engineering
- Include expanded features: raw lagged variables plus rolling technical indicators (e.g., moving averages, rate of change).

### Explainability Depth
- Use native Random Forest feature importance for basic explainability in the MVP. SHAP integration is deferred for potential future enhancements.

### Claude's Discretion
- Code architecture and specific directory layout for scripts (e.g., `src/models/train.py`, `src/features/build_features.py`).
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Scope
- `.planning/PROJECT.md` — Core MVP scope restrictions.
- `.planning/REQUIREMENTS.md` — ML requirements (ML-01).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/data/ingest.py` — Pipeline from Phase 1 (`load_and_clean_data(filepath)`). Outputs clean, daily frequency, non-NaN Pandas DataFrames.

### Established Patterns
- Python standard library + Pandas + Scikit-learn (to be added mapping).

### Integration Points
- This phase imports from Phase 1 and outputs artifacts (models, metrics, importance CSVs) for Phase 3 (Knowledge Graph).
</code_context>

<specifics>
## Specific Ideas
- Generate standard evaluation metrics (RMSE, MAE, R²) and basic feature importance lists when evaluating the models.
</specifics>

<deferred>
## Deferred Ideas
- SHAP values for advanced XAI and explanations.
</deferred>

---

*Phase: 2-core-ml-modeling*
*Context gathered: 2026-03-19*
