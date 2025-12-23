from scipy import stats
from math import log, pow, sqrt, exp

cdf = stats.norm.cdf
pdf = stats.norm.pdf


def calculate_d1(
    s: float,
    k: float,
    r: float,
    t: float,
    v: float
) -> float:
    """Calculate option D1 value"""
    d1: float = (log(s / k) + (0.5 * pow(v, 2)) * t) / (v * sqrt(t))
    return d1


def calculate_price(
    s: float,
    k: float,
    r: float,
    t: float,
    v: float,
    cp: int,
    d1: float = 0.0
) -> float:
    """Calculate option price"""
    # Return option space value if volatility not positive
    if v <= 0:
        return max(0, cp * (s - k)) * exp(-r * t)

    if not d1:
        d1 = calculate_d1(s, k, r, t, v)
    d2: float = d1 - v * sqrt(t)

    price: float = cp * (s * cdf(cp * d1) - k * cdf(cp * d2)) * exp(-r * t)
    return price


def calculate_delta(
    s: float,
    k: float,
    r: float,
    t: float,
    v: float,
    cp: int,
    d1: float = 0.0
) -> float:
    """Calculate option delta"""
    if v <= 0:
        return 0

    if not d1:
        d1 = calculate_d1(s, k, r, t, v)

    delta: float = cp * exp(-r * t) * cdf(cp * d1)
    return delta


def calculate_gamma(
    s: float,
    k: float,
    r: float,
    t: float,
    v: float,
    d1: float = 0.0
) -> float:
    """Calculate option gamma"""
    if v <= 0:
        return 0

    if not d1:
        d1 = calculate_d1(s, k, r, t, v)

    gamma: float = exp(-r * t) * pdf(d1) / (s * v * sqrt(t))
    return gamma


def calculate_theta(
    s: float,
    k: float,
    r: float,
    t: float,
    v: float,
    cp: int,
    d1: float = 0.0
) -> float:
    """Calculate option theta"""
    if v <= 0:
        return 0

    if not d1:
        d1 = calculate_d1(s, k, r, t, v)
    d2: float = d1 - v * sqrt(t)

    theta: float = -s * exp(-r * t) * pdf(d1) * v / (2 * sqrt(t)) \
        - cp * r * s * exp(-r * t) * cdf(cp * d1) \
        + cp * r * k * exp(-r * t) * cdf(cp * d2)
    return theta


def calculate_vega(
    s: float,
    k: float,
    r: float,
    t: float,
    v: float,
    d1: float = 0.0
) -> float:
    """Calculate option vega"""
    if v <= 0:
        return 0

    if not d1:
        d1 = calculate_d1(s, k, r, t, v)

    vega: float = s * exp(-r * t) * pdf(d1) * sqrt(t)
    return vega


def calculate_greeks(
    s: float,
    k: float,
    r: float,
    t: float,
    v: float,
    cp: int
) -> tuple[float, float, float, float, float]:
    """Calculate option price and greeks"""
    d1: float = calculate_d1(s, k, r, t, v)
    price: float = calculate_price(s, k, r, t, v, cp, d1)
    delta: float = calculate_delta(s, k, r, t, v, cp, d1)
    gamma: float = calculate_gamma(s, k, r, t, v, d1)
    theta: float = calculate_theta(s, k, r, t, v, cp, d1)
    vega: float = calculate_vega(s, k, r, t, v, d1)
    return price, delta, gamma, theta, vega


def calculate_impv(
    price: float,
    s: float,
    k: float,
    r: float,
    t: float,
    cp: int
) -> float:
    """Calculate option implied volatility"""
    # Check option price must be positive
    if price <= 0:
        return 0

    # Check if option price meets minimum value (exercise value)
    meet: bool = price > cp * (s - k) * exp(-r * t)

    # If minimum value not met, return 0
    if not meet:
        return 0

    # Calculate implied volatility with Newton's method
    # Smart initial guess based on moneyness
    if cp == 1:
        moneyness: float = s / k
    else:
        moneyness = k / s

    v_base: float = (price / s) / sqrt(t) * 2.5

    # Adjust based on moneyness
    if moneyness < 0.9:
        adjustment: float = (1 + (1 - moneyness) * 20)
    elif moneyness > 1.15:
        adjustment = max(0.6, 1 - (moneyness - 1) * 0.2)
    else:
        adjustment = 1.0

    v: float = v_base * adjustment
    v = min(max(v, 0.2), 5.0)

    for _i in range(100):
        # Calculate option price and vega with current guess
        p: float = calculate_price(s, k, r, t, v, cp)
        vega: float = calculate_vega(s, k, r, t, v)

        # Break loop if vega too close to 0
        if not vega or abs(vega) < 1e-10:
            break

        # Calculate error value
        dx: float = (price - p) / vega

        # Check if error value meets requirement
        if abs(dx) < 0.00001:
            break

        # Limit step size
        dx = max(-0.5, min(0.5, dx))

        # Calculate guessed implied volatility of next round
        v_new: float = v + dx
        if v_new <= 0:
            v_new = v * 0.5
        v = min(v_new, 10.0)

    # Final check
    if v <= 0:
        return 0

    # Round to 4 decimal places
    v = round(v, 4)

    return v
