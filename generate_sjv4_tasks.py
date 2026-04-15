#!/usr/bin/env python3
"""
generate_sjv4_tasks.py
======================
Generates 20 tasks specifically designed to show semantic corruption
from stale context. All tasks require cumulative intermediate state —
meaning a stale agent (missing steps 3-10) will produce outputs that
contradict what fresh agents have accumulated.

OUTPUT: sjv4_tasks.json (pass to run_sjv4_parallel.py --tasks-file)

USAGE:
    python3 generate_sjv4_tasks.py --output sjv4_tasks.json
    python3 run_sjv4_parallel.py --tasks-file sjv4_tasks.json --n-tasks 20 ...
"""

import json
import argparse

# 20 tasks requiring cumulative state
# Key design principle: initial_content is a running state document
# that accumulates entries at each step. A stale agent missing steps 5-15
# will propose additions that duplicate, contradict, or are incompatible
# with the accumulated state.

TASKS = [
    # ── SymPy — algebraic derivation ─────────────────────────────────────────
    {
        "task_id":   "sympy__solver-positive-assumption",
        "domain":    "sympy_solver",
        "problem_statement": (
            "Fix SymPy solve() dropping solutions with positive=True assumption. "
            "Track the derivation state: step 1 identifies root cause in assumptions.py, "
            "step 2 proposes the assumptions check fix, step 3 proposes the "
            "solution filter fix, step 4 adds tests. "
            "Each step MUST reference the specific line numbers and method names "
            "discovered in prior steps. A stale agent missing steps 2-4 will "
            "propose fixes that contradict or duplicate already-chosen approaches."
        ),
        "initial_content": (
            "DERIVATION STATE — solve() positive assumption fix:\n"
            "  Root cause: [not yet identified]\n"
            "  Affected method: [unknown]\n"
            "  Proposed fix: [none yet]\n"
            "  Tests added: []\n"
            "  Constraints found: []\n"
            "  Next step: identify root cause in assumptions.py"
        ),
    },
    {
        "task_id":   "sympy__matrix-eigenvals-sparse",
        "domain":    "sympy_matrix",
        "problem_statement": (
            "Fix SymPy eigenvals() for sparse symbolic matrices. "
            "Track the computation: step 1 computes characteristic polynomial det(A-λI), "
            "step 2 factors it, step 3 solves for eigenvalues, step 4 verifies. "
            "Each step records exact symbolic expressions. A stale agent "
            "will propose a factorisation that contradicts the polynomial found in step 1."
        ),
        "initial_content": (
            "EIGENVALUE COMPUTATION — sparse symbolic matrix A:\n"
            "  Matrix A: [[a, b], [c, d]] (symbolic entries)\n"
            "  Characteristic polynomial p(λ): [not computed]\n"
            "  Factored form: [unknown]\n"
            "  Eigenvalues found: []\n"
            "  Verification status: pending\n"
            "  Approach chosen: [none yet]"
        ),
    },
    {
        "task_id":   "sympy__integral-convergence",
        "domain":    "sympy_solver",
        "problem_statement": (
            "Fix SymPy integrate() failing for improper integrals with assumptions. "
            "Agents must track: which convergence conditions have been checked, "
            "which integration techniques have been tried and rejected, "
            "and the current best partial result. "
            "A stale agent will retry techniques already proven to fail."
        ),
        "initial_content": (
            "INTEGRATION PROGRESS — integrate(f(x), (x, 0, oo)):\n"
            "  Techniques tried: []\n"
            "  Techniques failed: []\n"
            "  Current best result: [none]\n"
            "  Convergence conditions checked: []\n"
            "  Assumptions applied: []\n"
            "  Remaining approaches: [by_parts, substitution, residue, series]"
        ),
    },
    # ── Django — migration graph ──────────────────────────────────────────────
    {
        "task_id":   "django__migration-squasher-circular",
        "domain":    "django_migration",
        "problem_statement": (
            "Fix Django migration squasher for circular dependencies. "
            "Agents must track the exact migration graph being built: "
            "which migrations have been processed, which dependencies added, "
            "which cycles detected and resolved. "
            "A stale agent missing steps 3-8 of the squash will add migrations "
            "that create new cycles or duplicate already-resolved dependencies."
        ),
        "initial_content": (
            "SQUASH STATE — migration graph:\n"
            "  Processed: []\n"
            "  Pending: [0001, 0002, 0003, 0004, 0005, 0006, 0007, 0008]\n"
            "  Dependencies added to squashed: []\n"
            "  Cycles detected: []\n"
            "  Cycles resolved: []\n"
            "  Squashed file: [not yet written]"
        ),
    },
    {
        "task_id":   "django__queryset-annotation-ordering",
        "domain":    "django_queryset",
        "problem_statement": (
            "Fix Django queryset ordering with annotated fields and select_related. "
            "Track: which SQL clauses have been fixed, which test cases pass, "
            "which edge cases remain. "
            "A stale agent will fix clauses already handled or introduce "
            "contradictory ORDER BY logic."
        ),
        "initial_content": (
            "FIX PROGRESS — queryset ordering:\n"
            "  SQL clauses examined: []\n"
            "  Clauses fixed: []\n"
            "  Tests passing: []\n"
            "  Tests failing: [test_order_by_related, test_select_related_order]\n"
            "  Current approach: [not chosen]\n"
            "  Remaining edge cases: [null_ordering, multi_table_join, reverse_fk]"
        ),
    },
    {
        "task_id":   "django__admin-permissions-accumulation",
        "domain":    "django_admin",
        "problem_statement": (
            "Fix Django admin per-object permissions for custom actions. "
            "Track which permission checks have been added to which methods, "
            "and which admin classes have been patched. "
            "A stale agent will patch already-patched methods or miss "
            "the method ordering established in prior steps."
        ),
        "initial_content": (
            "PATCH STATE — admin permissions:\n"
            "  Methods patched: []\n"
            "  Admin classes updated: []\n"
            "  Permission checks added: []\n"
            "  Tests written: []\n"
            "  Remaining methods: [get_action_choices, response_action, "
            "changelist_view, has_action_permission]"
        ),
    },
    # ── Astropy — header/registry accumulation ────────────────────────────────
    {
        "task_id":   "astropy__fits-hierarch-parser",
        "domain":    "astropy_fits",
        "problem_statement": (
            "Fix Astropy FITS HIERARCH keyword parser. "
            "Agents build a keyword registry incrementally: each step adds "
            "one new keyword type with its validation rule. "
            "A stale agent missing steps 3-10 will add duplicate keywords "
            "or define conflicting validation rules for already-registered types."
        ),
        "initial_content": (
            "KEYWORD REGISTRY STATE:\n"
            "  Registered keyword types: []\n"
            "  Validation rules defined: []\n"
            "  Conflict checks: []\n"
            "  Parse error handlers: []\n"
            "  Remaining types to add: [HIERARCH, CONTINUE, COMMENT, "
            "HISTORY, BLANK, END, SIMPLE, BITPIX]"
        ),
    },
    {
        "task_id":   "astropy__wcs-projection-edge-cases",
        "domain":    "astropy_wcs",
        "problem_statement": (
            "Fix Astropy WCS ZEA projection near poles. "
            "Agents track which boundary conditions have been added, "
            "which test projections have been verified, and the current "
            "numerical tolerance settings. "
            "A stale agent will add boundary checks that conflict with "
            "tolerance values set in prior steps."
        ),
        "initial_content": (
            "WCS FIX STATE — ZEA projection:\n"
            "  Boundary conditions added: []\n"
            "  Tolerance values set: {}\n"
            "  Projections tested: []\n"
            "  Projections failing: [ZEA_north_pole, ZEA_south_pole, ZEA_equator]\n"
            "  Current numerical approach: [not chosen]"
        ),
    },
    {
        "task_id":   "astropy__units-composite-conversion",
        "domain":    "astropy_units",
        "problem_statement": (
            "Fix Astropy composite unit conversion via non-SI intermediates. "
            "Agents build the conversion graph incrementally: each step adds "
            "one conversion path. A stale agent will add paths that "
            "create cycles or duplicate existing conversions, "
            "breaking the graph traversal algorithm."
        ),
        "initial_content": (
            "CONVERSION GRAPH STATE:\n"
            "  Paths added: []\n"
            "  Cycles detected: []\n"
            "  Equivalencies registered: []\n"
            "  Remaining conversions: [photon_flux, spectral_density, "
            "surface_brightness, energy_flux]\n"
            "  Graph traversal algorithm: [not chosen]"
        ),
    },
    # ── requests — session state ──────────────────────────────────────────────
    {
        "task_id":   "requests__session-auth-redirect",
        "domain":    "requests_session",
        "problem_statement": (
            "Fix requests session auth header stripping on cross-domain redirect. "
            "Agents track: which redirect scenarios have been handled, "
            "which headers are on the whitelist, and the current policy. "
            "A stale agent will add headers already on the whitelist "
            "or propose a policy that contradicts the one chosen in step 3."
        ),
        "initial_content": (
            "AUTH REDIRECT FIX STATE:\n"
            "  Redirect scenarios handled: []\n"
            "  Header whitelist: []\n"
            "  Current policy: [not set]\n"
            "  Methods patched: []\n"
            "  Remaining scenarios: [cross_domain, https_to_http, "
            "301_permanent, 302_found, 307_temporary]"
        ),
    },
    {
        "task_id":   "requests__session-cookie-https",
        "domain":    "requests_session",
        "problem_statement": (
            "Fix requests session cookie handling with HTTPS redirects. "
            "Track which cookie attributes have been preserved in the redirect chain, "
            "which security checks have been added. "
            "A stale agent will re-add already-handled attributes or "
            "contradict the Secure flag policy set in step 2."
        ),
        "initial_content": (
            "COOKIE REDIRECT FIX STATE:\n"
            "  Cookie attributes handled: []\n"
            "  Security checks added: []\n"
            "  Secure flag policy: [not set]\n"
            "  SameSite policy: [not set]\n"
            "  Test cases written: []\n"
            "  Remaining: [Secure, HttpOnly, SameSite, Domain, Path, Expires]"
        ),
    },
    # ── scikit-learn — estimator API ──────────────────────────────────────────
    {
        "task_id":   "sklearn__clone-kwargs-estimator",
        "domain":    "scikit_estimator",
        "problem_statement": (
            "Fix scikit-learn clone() for estimators with **kwargs in __init__. "
            "Agents track: which __init__ signature patterns have been handled, "
            "which parameter extraction strategies have been tried. "
            "A stale agent will re-try strategies already proven to fail "
            "or handle already-covered signature patterns."
        ),
        "initial_content": (
            "CLONE FIX STATE:\n"
            "  __init__ patterns handled: []\n"
            "  Strategies tried: []\n"
            "  Strategies failed: []\n"
            "  Current best approach: [none]\n"
            "  Remaining patterns: [args_only, kwargs_only, "
            "mixed_args_kwargs, default_factory, class_level_defaults]"
        ),
    },
    {
        "task_id":   "sklearn__pipeline-set-params",
        "domain":    "scikit_estimator",
        "problem_statement": (
            "Fix scikit-learn set_params() for nested pipeline estimators. "
            "Agents build a parameter routing table incrementally. "
            "A stale agent missing the routing entries from steps 3-7 "
            "will add duplicate routes or create routing conflicts."
        ),
        "initial_content": (
            "PARAMETER ROUTING TABLE:\n"
            "  Routes defined: []\n"
            "  Conflicts found: []\n"
            "  Conflicts resolved: []\n"
            "  Nested estimators handled: []\n"
            "  Remaining: [Pipeline, FeatureUnion, ColumnTransformer, "
            "GridSearchCV, CalibratedClassifierCV]"
        ),
    },
    # ── More SymPy ────────────────────────────────────────────────────────────
    {
        "task_id":   "sympy__series-expansion-order",
        "domain":    "sympy_solver",
        "problem_statement": (
            "Fix SymPy series() incorrect expansion order for composite functions. "
            "Agents track: which expansion terms have been computed, "
            "which order O() terms have been verified. "
            "A stale agent will recompute already-verified terms "
            "or claim wrong O() order for terms computed in prior steps."
        ),
        "initial_content": (
            "SERIES EXPANSION STATE — f(x) = sin(exp(x)) around x=0:\n"
            "  Terms computed: []\n"
            "  Terms verified: []\n"
            "  Current order: [not set]\n"
            "  O() boundary: [not set]\n"
            "  Remaining terms: [x^0, x^1, x^2, x^3, x^4, x^5, x^6]"
        ),
    },
    {
        "task_id":   "sympy__matrix-determinant-rationals",
        "domain":    "sympy_matrix",
        "problem_statement": (
            "Fix SymPy det() for large symbolic matrices with Rational coefficients. "
            "Agents use Bareiss algorithm: each step eliminates one row. "
            "The intermediate pivot values from prior steps MUST be used correctly. "
            "A stale agent missing pivots from steps 2-5 will compute "
            "wrong determinant from incorrect intermediate state."
        ),
        "initial_content": (
            "BAREISS ELIMINATION STATE — 6×6 matrix:\n"
            "  Steps completed: 0/5\n"
            "  Current matrix: [original, no elimination done]\n"
            "  Pivot history: []\n"
            "  Intermediate determinant sign: [unknown]\n"
            "  Remaining eliminations: [step1, step2, step3, step4, step5]"
        ),
    },
    # ── More Django ───────────────────────────────────────────────────────────
    {
        "task_id":   "django__orm-subquery-annotation",
        "domain":    "django_queryset",
        "problem_statement": (
            "Fix Django ORM subquery annotation with Exists() and OuterRef(). "
            "Track which SQL generation paths have been fixed and tested. "
            "A stale agent will re-fix already-handled paths "
            "or introduce SQL that conflicts with JOIN strategy set in step 4."
        ),
        "initial_content": (
            "ORM FIX STATE — subquery annotation:\n"
            "  SQL paths fixed: []\n"
            "  JOIN strategy: [not chosen]\n"
            "  Subquery types handled: []\n"
            "  Tests passing: []\n"
            "  Remaining: [Exists, Subquery, OuterRef_FK, OuterRef_M2M, "
            "Correlated_subquery, Uncorrelated_subquery]"
        ),
    },
    {
        "task_id":   "django__forms-validation-order",
        "domain":    "django_admin",
        "problem_statement": (
            "Fix Django form field validation order for interdependent fields. "
            "Agents build a validation dependency graph: each step adds one edge. "
            "A stale agent will add edges that create cycles in the graph "
            "built by fresh agents in steps 3-9, breaking topological sort."
        ),
        "initial_content": (
            "VALIDATION GRAPH STATE:\n"
            "  Nodes (fields): [username, email, password, confirm_password, "
            "phone, address, zip_code]\n"
            "  Dependency edges: []\n"
            "  Cycles detected: []\n"
            "  Topological order: [not computed]\n"
            "  Validation methods added: []"
        ),
    },
    # ── More Astropy ──────────────────────────────────────────────────────────
    {
        "task_id":   "astropy__table-column-accumulation",
        "domain":    "astropy_fits",
        "problem_statement": (
            "Fix Astropy Table column type inference for mixed FITS extensions. "
            "Agents track which column types have been registered and their "
            "precedence order. A stale agent missing precedence rules set "
            "in steps 4-8 will register conflicting types for the same column."
        ),
        "initial_content": (
            "COLUMN TYPE REGISTRY:\n"
            "  Types registered: []\n"
            "  Precedence order: []\n"
            "  Conflicts resolved: []\n"
            "  Remaining types: [int16, int32, int64, float32, float64, "
            "bool, string, bytes, complex64, complex128]"
        ),
    },
    {
        "task_id":   "astropy__coordinates-frame-chain",
        "domain":    "astropy_wcs",
        "problem_statement": (
            "Fix Astropy coordinate frame transformation chain. "
            "Agents build the transformation graph incrementally. "
            "Each step adds one frame-to-frame transformation. "
            "A stale agent missing transformations from steps 3-9 "
            "will create disconnected graph components or duplicate edges."
        ),
        "initial_content": (
            "FRAME TRANSFORMATION GRAPH:\n"
            "  Frames registered: [ICRS]\n"
            "  Transformations added: []\n"
            "  Graph components: 1 (disconnected)\n"
            "  Remaining frames: [FK5, FK4, Galactic, GalacticLSR, "
            "CIRS, GCRS, ITRS, AltAz, HADec]\n"
            "  Required: fully connected graph"
        ),
    },
    {
        "task_id":   "astropy__spectrum-wavelength-grid",
        "domain":    "astropy_units",
        "problem_statement": (
            "Fix Astropy Spectrum1D wavelength grid resampling. "
            "Agents track which resampling methods have been implemented "
            "and their validated wavelength ranges. "
            "A stale agent will implement methods already done "
            "or claim wrong validated ranges from prior steps."
        ),
        "initial_content": (
            "RESAMPLING STATE:\n"
            "  Methods implemented: []\n"
            "  Validated ranges: {}\n"
            "  Current grid: [not set]\n"
            "  Interpolation strategy: [not chosen]\n"
            "  Remaining methods: [linear, spline, flux_conserving, "
            "weighted_mean, median, gaussian_kernel]"
        ),
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sjv4_tasks.json")
    args = parser.parse_args()

    print(f"Generated {len(TASKS)} tasks for SJ-v4")
    print()
    print("Task domains:")
    from collections import Counter
    counts = Counter(t["domain"] for t in TASKS)
    for domain, n in sorted(counts.items()):
        print(f"  {domain:<30}: {n} tasks")
    print()
    print("All tasks use cumulative state design:")
    print("  - Initial content is a running state document")
    print("  - Each step adds to or modifies the state")
    print("  - Stale agent missing steps 3-10 produces contradictory output")
    print()

    with open(args.output, "w") as f:
        json.dump(TASKS, f, indent=2)
    print(f"Saved: {args.output}")
    print()
    print("Run with:")
    print(f"  python3 run_sjv4_parallel.py \\")
    print(f"      --tasks-file {args.output} \\")
    print(f"      --n-tasks 20 \\")
    print(f"      --n-runs 25 \\")
    print(f"      --n-steps 20 \\")
    print(f"      --injection-step 5 \\")
    print(f"      --base-port 7010 \\")
    print(f"      --sbus-bin ~/sbus/target/release/sbus \\")
    print(f"      --output results/sj_v4_results.csv")


if __name__ == "__main__":
    main()