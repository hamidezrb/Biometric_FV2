"""

ISO/IEC 29794-9 — Section 5.3 unified vascular image quality score.



Combines individual quality components Q1..Q9 into a single integer score in [0, 100].

"""



from __future__ import annotations



import math

from typing import Mapping, Optional, Sequence, Union



N_COMPONENTS = 9

SCORE_MAX = 100



# Clause 5.2 component order for Formula (17): i = 1..9

COMPONENT_KEYS: tuple[str, ...] = tuple(f"Q{i}" for i in range(1, N_COMPONENTS + 1))



# Human-readable labels (Clause 5.2 subclauses) — order matches COMPONENT_KEYS.

COMPONENT_LABELS: tuple[str, ...] = (

    "effective_area",  # 5.2.1

    "offset_complement",  # 5.2.2

    "contrast",  # 5.2.3

    "equivalent_number_of_looks",  # 5.2.4

    "information_entropy",  # 5.2.5

    "sharpness",  # 5.2.6

    "brightness_uniformity",  # 5.2.7

    "total_vascular_length",  # 5.2.8

    "number_of_feature_points",  # 5.2.9

)



# TODO: Draft 5.3 references calibrated α_i in the ISO reference implementation

# (https://github.com/TBD). No official coefficients are bundled in this repository.

# PLACEHOLDER ONLY — equal α_i = 1/N_Q (geometric mean of normalized components).

# Do not treat as normative until the reference implementation values are imported.

DEFAULT_POWER_COEFFICIENTS: tuple[float, ...] = tuple(

    1.0 / N_COMPONENTS for _ in range(N_COMPONENTS)

)

COEFFICIENTS_ARE_PLACEHOLDER = True

UNIFIED_WEIGHT_LABEL = (
    "experimental equal-weight baseline (alpha_i = 1/9 each; NOT normative ISO coefficients)"
)


def unified_weight_description() -> str:
    """Human-readable description of unified-score weighting mode."""
    if COEFFICIENTS_ARE_PLACEHOLDER:
        return UNIFIED_WEIGHT_LABEL
    return "calibrated ISO reference implementation coefficients"





QiInput = Union[int, float, None]

QiSequence = Sequence[QiInput]

QiMapping = Mapping[str, QiInput]





def _normalize_qi(value: QiInput) -> float:

    """Return Qi in [0, 100]; missing or invalid values become 0."""

    if value is None:

        return 0.0

    try:

        q = float(value)

    except (TypeError, ValueError):

        return 0.0

    if math.isnan(q) or math.isinf(q):

        return 0.0

    return max(0.0, min(float(SCORE_MAX), q))





def _qi_list_from_input(qi_scores: Union[QiSequence, QiMapping]) -> list[float]:

    if isinstance(qi_scores, Mapping):

        return [_normalize_qi(qi_scores.get(k)) for k in COMPONENT_KEYS]

    values = list(qi_scores)

    if len(values) != N_COMPONENTS:

        raise ValueError(

            f"Expected {N_COMPONENTS} quality components, got {len(values)}."

        )

    return [_normalize_qi(v) for v in values]





def _validate_coefficients(power_coefficients: Sequence[float]) -> tuple[float, ...]:

    coeffs = tuple(float(w) for w in power_coefficients)

    if len(coeffs) != N_COMPONENTS:

        raise ValueError(

            f"Expected {N_COMPONENTS} power coefficients, got {len(coeffs)}."

        )

    return coeffs





def format_power_coefficients(coefficients: Sequence[float]) -> str:

    """Human-readable α1..α9 listing for reports."""

    return ", ".join(f"a{i}={float(c):.6g}" for i, c in enumerate(coefficients, start=1))





def compute_formula_17(

    qi: Sequence[float],

    power_coefficients: Sequence[float],

) -> float:

    """

    Formula (17) before rounding/clamp:



    100 * prod((Qi/100) ** α_i) for i = 1..N_Q, N_Q = 9.

    """

    product_term = 1.0

    for q, alpha in zip(qi, power_coefficients):

        if alpha == 0.0:

            continue

        product_term *= (q / SCORE_MAX) ** alpha

    return SCORE_MAX * product_term





def evaluate_unified_quality(

    qi_scores: Union[QiSequence, QiMapping],

    power_coefficients: Optional[Sequence[float]] = None,

) -> dict[str, object]:

    """

    ISO/IEC 29794-9 Section 5.3 — unified quality score and coefficients used.



    Returns unified_quality_score (int), power_coefficients (9-tuple),

    and coefficients_are_placeholder (bool).

    """

    qi = _qi_list_from_input(qi_scores)

    using_default = power_coefficients is None

    coeffs = _validate_coefficients(

        power_coefficients if power_coefficients is not None else DEFAULT_POWER_COEFFICIENTS

    )



    # Veto: any Qi = 0 with α_i > 0 drives the product (and Q) to 0.

    for q, alpha in zip(qi, coeffs):

        if q == 0.0 and alpha > 0.0:

            return {

                "unified_quality_score": 0,

                "power_coefficients": coeffs,

                "coefficients_are_placeholder": using_default and COEFFICIENTS_ARE_PLACEHOLDER,

            }



    raw = compute_formula_17(qi, coeffs)

    unified = max(0, min(SCORE_MAX, int(round(raw))))

    return {

        "unified_quality_score": unified,

        "power_coefficients": coeffs,

        "coefficients_are_placeholder": using_default and COEFFICIENTS_ARE_PLACEHOLDER,

    }





def compute_unified_quality_score(

    qi_scores: Union[QiSequence, QiMapping],

    power_coefficients: Optional[Sequence[float]] = None,

) -> int:

    """

    ISO/IEC 29794-9 Section 5.3 — unified quality score (integer only).



    Q = ROUND(100 * prod((Qi/100)^α_i)), N_Q = 9.



    Veto: if any Qi is 0 and its α_i > 0, the unified score is 0.

    Each Qi is clamped to [0, 100]; missing or invalid Qi are treated as 0.

    """

    return int(

        evaluate_unified_quality(qi_scores, power_coefficients)["unified_quality_score"]

    )


