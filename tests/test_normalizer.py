import time
import pytest
from backend.normalizer import PropertyNormalizer


# ==========================================
# 1. CURRENCY NORMALIZATION TEST MATRIX (12 Cases)
# ==========================================
@pytest.mark.parametrize("raw_input, expected_pkr", [
    # Standard compounds
    ("2 crore 35 lac", 23_500_000),
    ("1 crore 15 lakh", 11_500_000),
    ("85 lacs", 8_500_000),
    ("50 lakh", 5_000_000),
    ("1.5 crore", 15_000_000),
    ("4.5 cr", 45_000_000),
    ("2 cr 10 lakh", 21_000_000),
    ("1 arab", 1_000_000_000),
    ("9500000", 9_500_000),  # Raw digits
    
    # Colloquial South Asian fractions
    ("dedh crore", 15_000_000),
    ("dhayi lac", 250_000),
    ("paunay do crore", 17_500_000),
])
def test_currency_normalization(raw_input, expected_pkr):
    result = PropertyNormalizer.normalize_price(raw_input)
    assert result == expected_pkr, f"Failed for '{raw_input}': got {result}, expected {expected_pkr}"


# ==========================================
# 2. LAND AREA NORMALIZATION TEST MATRIX (9 Cases)
# ==========================================
@pytest.mark.parametrize("raw_input, expected_sqft", [
    # LDA standard: 225 sqft/Marla, 4500 sqft/Kanal
    ("1 kanal", 4500.0),
    ("10 marla", 2250.0),
    ("5 marla", 1125.0),
    ("20 marla", 4500.0),
    ("0.5 kanal", 2250.0),
    ("2 kanal 5 marla", 10125.0),  # (2 * 4500) + (5 * 225)
    
    # Gaz / Square Yards (9 sqft)
    ("500 gaz", 4500.0),
    ("100 sqft", 100.0),
    ("150 yard", 1350.0),
])
def test_area_normalization(raw_input, expected_sqft):
    result = PropertyNormalizer.normalize_area(raw_input)
    assert result == expected_sqft, f"Failed for '{raw_input}': got {result}, expected {expected_sqft}"


# ==========================================
# 3. EMPTY & MALFORMED INPUT GRACEFUL HANDLING
# ==========================================
@pytest.mark.parametrize("empty_input", ["", "   ", None])
def test_empty_inputs_return_none(empty_input):
    assert PropertyNormalizer.normalize_price(empty_input) is None
    assert PropertyNormalizer.normalize_area(empty_input) is None


# ==========================================
# 4. CHAOS TEST 5.1: Regular Expression Backtracking Attack (ReDoS)
# ==========================================
def test_redos_backtracking_boundary():
    """
    Simulates an adversarial ReDoS payload: 2,000 repetitions of whitespace and terms.
    Asserts linear evaluation time (O(N)) completing in strictly < 15ms.
    """
    adversarial_payload = ("1 crore " * 2000) + "lakh"
    
    start_time = time.perf_counter()
    result = PropertyNormalizer.normalize_price(adversarial_payload)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Must complete in under 15 milliseconds without hanging worker threads
    assert elapsed_ms < 15.0, f"ReDoS vulnerability detected! Evaluation took {elapsed_ms:.2f}ms (> 15ms target)"
    assert result is not None
    print(f"\n[+] ReDoS Test Passed: Evaluated 2,000 adversarial repetitions in {elapsed_ms:.2f}ms")


# ==========================================
# 5. CHAOS TEST 5.2: Colloquial Vernacular Quantifiers
# ==========================================
def test_colloquial_vernacular_quantifiers():
    """
    Verifies that informal South Asian fractional vocabulary resolves deterministically.
    """
    assert PropertyNormalizer.normalize_price("dedh crore") == 15_000_000
    assert PropertyNormalizer.normalize_price("dhayi lac") == 250_000
    assert PropertyNormalizer.normalize_price("sawa crore") == 12_500_000
    assert PropertyNormalizer.normalize_price("paunay do crore") == 17_500_000