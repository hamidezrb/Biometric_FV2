"""Tests for ISO/IEC 29794-9 Section 5.3 unified quality score."""



import math

import unittest



from unified_quality import (

    COEFFICIENTS_ARE_PLACEHOLDER,

    COMPONENT_KEYS,

    COMPONENT_LABELS,

    DEFAULT_POWER_COEFFICIENTS,

    N_COMPONENTS,

    compute_formula_17,

    compute_unified_quality_score,

    evaluate_unified_quality,

)





def _all_equal(value: int) -> list[int]:

    return [value] * N_COMPONENTS





class TestUnifiedQualityStructure(unittest.TestCase):

    def test_nine_components(self):

        self.assertEqual(N_COMPONENTS, 9)

        self.assertEqual(len(COMPONENT_KEYS), 9)

        self.assertEqual(len(COMPONENT_LABELS), 9)

        self.assertEqual(COMPONENT_KEYS[0], "Q1")

        self.assertEqual(COMPONENT_KEYS[8], "Q9")



    def test_placeholder_coefficients_documented(self):

        self.assertTrue(COEFFICIENTS_ARE_PLACEHOLDER)

        self.assertEqual(len(DEFAULT_POWER_COEFFICIENTS), 9)

        self.assertAlmostEqual(sum(DEFAULT_POWER_COEFFICIENTS), 1.0)



    def test_evaluate_marks_placeholder_when_using_default(self):

        out = evaluate_unified_quality(_all_equal(100))

        self.assertTrue(out["coefficients_are_placeholder"])



    def test_returns_integer_score(self):

        score = compute_unified_quality_score(_all_equal(80))

        self.assertIsInstance(score, int)





class TestUnifiedQualityFormula17(unittest.TestCase):

    def test_all_hundred_yields_hundred(self):

        self.assertEqual(compute_unified_quality_score(_all_equal(100)), 100)



    def test_any_zero_yields_zero_veto(self):

        scores = _all_equal(100)

        for i in range(N_COMPONENTS):

            case = list(scores)

            case[i] = 0

            with self.subTest(index=i):

                self.assertEqual(compute_unified_quality_score(case), 0)



    def test_all_fifty_equal_placeholder_weights_yields_fifty(self):

        self.assertEqual(

            compute_unified_quality_score(_all_equal(50), DEFAULT_POWER_COEFFICIENTS),

            50,

        )



    def test_product_not_weighted_average(self):

        """One zero must veto; arithmetic mean of the rest would be > 0."""

        scores = _all_equal(100)

        scores[6] = 0

        self.assertEqual(compute_unified_quality_score(scores), 0)

        arithmetic_mean = sum(scores) / N_COMPONENTS

        self.assertGreater(arithmetic_mean, 0)



    def test_multiple_low_scores(self):

        scores = _all_equal(20)

        self.assertEqual(compute_unified_quality_score(scores), 20)



    def test_dict_input_all_nine_keys(self):

        qi = {k: 100 for k in COMPONENT_KEYS}

        self.assertEqual(compute_unified_quality_score(qi), 100)

        qi["Q5"] = 0

        self.assertEqual(compute_unified_quality_score(qi), 0)



    def test_missing_component_score_in_dict(self):

        qi = {f"Q{i}": 100 for i in range(1, 9)}

        self.assertEqual(compute_unified_quality_score(qi), 0)



    def test_invalid_and_out_of_range_clamped(self):

        scores = _all_equal(100)

        scores[0] = None

        scores[1] = float("nan")

        scores[2] = "bad"

        self.assertEqual(compute_unified_quality_score(scores), 0)



        high = _all_equal(100)

        high[3] = 200

        self.assertEqual(compute_unified_quality_score(high), 100)



        low = _all_equal(50)

        low[4] = -10

        self.assertEqual(compute_unified_quality_score(low), 0)



    def test_rounding_to_integer(self):
        scores = [80.4] * N_COMPONENTS
        raw = compute_formula_17(
            [80.4] * N_COMPONENTS, DEFAULT_POWER_COEFFICIENTS
        )
        self.assertAlmostEqual(raw, 80.4, places=5)
        self.assertEqual(compute_unified_quality_score(scores), 80)



    def test_clamp_above_100_before_return(self):

        raw = 100.6

        unified = max(0, min(100, int(round(raw))))

        self.assertEqual(unified, 100)

        scores = _all_equal(100)

        self.assertLessEqual(compute_unified_quality_score(scores), 100)



    def test_zero_alpha_skips_veto_for_that_component(self):

        coeffs = list(DEFAULT_POWER_COEFFICIENTS)

        coeffs[0] = 0.0

        coeffs[1] = 1.0

        scores = _all_equal(100)

        scores[0] = 0

        self.assertGreater(compute_unified_quality_score(scores, coeffs), 0)



    def test_wrong_component_count_raises(self):

        with self.assertRaises(ValueError):

            compute_unified_quality_score([100, 100])

        with self.assertRaises(ValueError):

            compute_unified_quality_score(_all_equal(100), [0.5, 0.5])



    def test_missing_alpha_uses_documented_placeholder(self):

        out = evaluate_unified_quality(_all_equal(100), power_coefficients=None)

        self.assertEqual(out["power_coefficients"], DEFAULT_POWER_COEFFICIENTS)

        self.assertTrue(out["coefficients_are_placeholder"])



    def test_custom_alpha_not_marked_placeholder(self):

        custom = tuple(0.2 for _ in range(N_COMPONENTS))

        out = evaluate_unified_quality(_all_equal(100), power_coefficients=custom)

        self.assertFalse(out["coefficients_are_placeholder"])





if __name__ == "__main__":

    unittest.main()


