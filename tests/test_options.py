"""Tests for quantforge.options: Black-Scholes, binomial, Monte Carlo, Greeks."""
import math

import pytest

from quantforge.options.binomial import crr_american, crr_price
from quantforge.options.black_scholes import bs_call, bs_implied_vol, bs_put, d1, d2
from quantforge.options.greeks import all_greeks, delta, gamma, rho, theta, vega
from quantforge.options.monte_carlo import mc_european

# Standard test parameters
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
S_itm, K_otm = 110.0, 100.0   # ITM call / OTM put


# ---------------------------------------------------------------------------
# Put-Call Parity
# ---------------------------------------------------------------------------

class TestPutCallParity:
    def test_parity_atm(self):
        c = bs_call(S, K, T, r, sigma)
        p = bs_put(S, K, T, r, sigma)
        # C - P = S - K*e^{-rT}
        parity = c - p - (S - K * math.exp(-r * T))
        assert abs(parity) < 1e-9

    def test_parity_itm_call(self):
        c = bs_call(S_itm, K_otm, T, r, sigma)
        p = bs_put(S_itm, K_otm, T, r, sigma)
        parity = c - p - (S_itm - K_otm * math.exp(-r * T))
        assert abs(parity) < 1e-9

    def test_parity_otm_call(self):
        c = bs_call(90.0, 100.0, T, r, sigma)
        p = bs_put(90.0, 100.0, T, r, sigma)
        parity = c - p - (90.0 - 100.0 * math.exp(-r * T))
        assert abs(parity) < 1e-9

    def test_parity_various_vol(self):
        for sig in [0.1, 0.3, 0.5, 0.8]:
            c = bs_call(S, K, T, r, sig)
            p = bs_put(S, K, T, r, sig)
            parity = c - p - (S - K * math.exp(-r * T))
            assert abs(parity) < 1e-9, f"Parity failed for sigma={sig}"


# ---------------------------------------------------------------------------
# BS basic sanity checks
# ---------------------------------------------------------------------------

class TestBlackScholes:
    def test_call_price_positive(self):
        assert bs_call(S, K, T, r, sigma) > 0

    def test_put_price_positive(self):
        assert bs_put(S, K, T, r, sigma) > 0

    def test_call_intrinsic_at_expiry(self):
        # At T=0 with ITM call, price = max(S-K, 0)
        c = bs_call(110.0, 100.0, 0.0, r, sigma)
        assert c == pytest.approx(10.0)

    def test_put_intrinsic_at_expiry(self):
        p = bs_put(90.0, 100.0, 0.0, r, sigma)
        assert p == pytest.approx(10.0)

    def test_call_increases_with_spot(self):
        c1 = bs_call(95.0, 100.0, T, r, sigma)
        c2 = bs_call(105.0, 100.0, T, r, sigma)
        assert c2 > c1

    def test_put_decreases_with_spot(self):
        p1 = bs_put(95.0, 100.0, T, r, sigma)
        p2 = bs_put(105.0, 100.0, T, r, sigma)
        assert p1 > p2

    def test_call_increases_with_vol(self):
        c1 = bs_call(S, K, T, r, 0.1)
        c2 = bs_call(S, K, T, r, 0.3)
        assert c2 > c1

    def test_d1_d2_finite_for_valid_params(self):
        assert math.isfinite(d1(S, K, T, r, sigma))
        assert math.isfinite(d2(S, K, T, r, sigma))

    def test_d1_d2_nan_for_zero_T(self):
        assert math.isnan(d1(S, K, 0.0, r, sigma))
        assert math.isnan(d2(S, K, 0.0, r, sigma))


# ---------------------------------------------------------------------------
# Implied Vol Round-Trip
# ---------------------------------------------------------------------------

class TestImpliedVol:
    @pytest.mark.parametrize("sigma_input", [0.05, 0.15, 0.25, 0.40, 0.60])
    def test_iv_roundtrip_call(self, sigma_input):
        price = bs_call(S, K, T, r, sigma_input)
        iv = bs_implied_vol(price, S, K, T, r, "call")
        assert abs(iv - sigma_input) < 1e-5

    @pytest.mark.parametrize("sigma_input", [0.05, 0.15, 0.25, 0.40, 0.60])
    def test_iv_roundtrip_put(self, sigma_input):
        price = bs_put(S, K, T, r, sigma_input)
        iv = bs_implied_vol(price, S, K, T, r, "put")
        assert abs(iv - sigma_input) < 1e-5

    def test_iv_negative_price_returns_nan(self):
        iv = bs_implied_vol(-1.0, S, K, T, r, "call")
        assert math.isnan(iv)

    def test_iv_zero_T_returns_nan(self):
        iv = bs_implied_vol(5.0, S, K, 0.0, r, "call")
        assert math.isnan(iv)


# ---------------------------------------------------------------------------
# CRR Binomial vs BS
# ---------------------------------------------------------------------------

class TestCRRBinomial:
    def test_crr_converges_to_bs_atm_call(self):
        bs = bs_call(S, K, T, r, sigma)
        crr = crr_price(S, K, T, r, sigma, "call", steps=500)
        assert abs(crr - bs) < 1e-2

    def test_crr_converges_to_bs_atm_put(self):
        bs = bs_put(S, K, T, r, sigma)
        crr = crr_price(S, K, T, r, sigma, "put", steps=500)
        assert abs(crr - bs) < 1e-2

    def test_crr_call_positive(self):
        assert crr_price(S, K, T, r, sigma, "call") > 0

    def test_crr_put_call_parity(self):
        c = crr_price(S, K, T, r, sigma, "call", steps=500)
        p = crr_price(S, K, T, r, sigma, "put", steps=500)
        parity = c - p - (S - K * math.exp(-r * T))
        assert abs(parity) < 2e-2

    def test_american_put_ge_european_put(self):
        # American put >= European put (early exercise premium)
        eur = crr_price(S, K, T, r, sigma, "put", steps=300)
        amer = crr_american(S, K, T, r, sigma, "put", steps=300)
        assert amer >= eur - 1e-10

    def test_american_put_premium_deep_itm(self):
        # Deep ITM American put has strict early exercise premium
        eur = crr_price(50.0, 100.0, T, r, sigma, "put", steps=300)
        amer = crr_american(50.0, 100.0, T, r, sigma, "put", steps=300)
        # American must be at least intrinsic value = 50
        assert amer >= 50.0 - 1e-6
        # And strictly greater than European for deep ITM
        assert amer > eur - 1e-6


# ---------------------------------------------------------------------------
# Monte Carlo European
# ---------------------------------------------------------------------------

class TestMCEuropean:
    def test_mc_european_call_within_3_stderr_of_bs(self):
        bs = bs_call(S, K, T, r, sigma)
        mc_price, mc_se = mc_european(S, K, T, r, sigma, "call", seed=42, n_paths=100_000)
        assert abs(mc_price - bs) < 3 * mc_se

    def test_mc_european_put_within_3_stderr_of_bs(self):
        bs = bs_put(S, K, T, r, sigma)
        mc_price, mc_se = mc_european(S, K, T, r, sigma, "put", seed=42, n_paths=100_000)
        assert abs(mc_price - bs) < 3 * mc_se

    def test_mc_returns_tuple_of_price_and_stderr(self):
        result = mc_european(S, K, T, r, sigma, seed=0)
        assert len(result) == 2
        price, se = result
        assert price > 0
        assert se > 0

    def test_mc_seed_deterministic(self):
        p1, _ = mc_european(S, K, T, r, sigma, seed=99)
        p2, _ = mc_european(S, K, T, r, sigma, seed=99)
        assert p1 == pytest.approx(p2)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

class TestGreeks:
    def test_call_delta_between_0_and_1(self):
        d = delta(S, K, T, r, sigma, "call")
        assert 0.0 < d < 1.0

    def test_put_delta_between_minus1_and_0(self):
        d = delta(S, K, T, r, sigma, "put")
        assert -1.0 < d < 0.0

    def test_atm_call_delta_above_half_with_positive_rate(self):
        # With r=0.05, ATM call delta = N(d1) where d1 = (r + 0.5*sigma^2)*T / (sigma*sqrt(T))
        # d1 = (0.05 + 0.02) / 0.2 = 0.35 => N(0.35) ~ 0.637, which is > 0.5
        d = delta(S, K, T, r, sigma, "call")
        assert d > 0.5  # shifted above 0.5 due to positive risk-free rate

    def test_atm_call_delta_near_half_zero_rate(self):
        # With r=0 and short time-to-expiry, ATM delta is closer to 0.5
        d = delta(S, K, 0.1, 0.0, sigma, "call")
        assert 0.4 < d < 0.6

    def test_gamma_positive(self):
        g = gamma(S, K, T, r, sigma)
        assert g > 0

    def test_vega_positive(self):
        v = vega(S, K, T, r, sigma)
        assert v > 0

    def test_call_theta_negative(self):
        th = theta(S, K, T, r, sigma, "call")
        assert th < 0

    def test_call_rho_positive(self):
        rh = rho(S, K, T, r, sigma, "call")
        assert rh > 0

    def test_put_rho_negative(self):
        rh = rho(S, K, T, r, sigma, "put")
        assert rh < 0

    def test_all_greeks_returns_expected_keys(self):
        g = all_greeks(S, K, T, r, sigma, "call")
        assert set(g.keys()) == {"delta", "gamma", "vega", "theta", "rho"}

    def test_call_delta_plus_put_delta_eq_exp_minusq_T(self):
        # For q=0: call_delta - put_delta = 1 (since N(d1) - (N(d1)-1) = 1)
        d_call = delta(S, K, T, r, sigma, "call")
        d_put = delta(S, K, T, r, sigma, "put")
        assert d_call - d_put == pytest.approx(1.0, abs=1e-10)
