from numpy import zeros, ndarray
from math import exp, sqrt
from typing import Tuple


DEFAULT_STEP = 15


def generate_tree(
    f: float,
    k: float,
    r: float,
    t: float,
    v: float,
    cp: int,
    n: int
) -> Tuple[ndarray, ndarray]:
    """Generate binomial tree for pricing American option."""
    dt: float = t / n
    u: float = exp(v * sqrt(dt))
    d: float = 1 / u
    a: float = 1
    tree_size: int = n + 1

    underlying_tree: ndarray = zeros((tree_size, tree_size))
    option_tree: ndarray = zeros((tree_size, tree_size))

    # Calculate risk neutral probability
    p: float = (a - d) / (u - d)
    p1: float = p / a
    p2: float = (1 - p) / a
    discount: float = exp(-r * dt)

    # Calculate underlying price tree
    underlying_tree[0, 0] = f

    for i in range(1, n + 1):
        underlying_tree[0, i] = underlying_tree[0, i - 1] * u
        for j in range(1, n + 1):
            underlying_tree[j, i] = underlying_tree[j - 1, i - 1] * d

    # Calculate option price tree
    for j in range(n + 1):
        option_tree[j, n] = max(0, cp * (underlying_tree[j, n] - k))

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_tree[j, i] = max(
                (p1 * option_tree[j, i + 1] + p2 * option_tree[j + 1, i + 1]) * discount,
                cp * (underlying_tree[j, i] - k),
                0
            )

    # Return both trees
    return option_tree, underlying_tree


def calculate_price(
    f: float,
    k: float,
    r: float,
    t: float,
    v: float,
    cp: int,
    n: int = DEFAULT_STEP
) -> float:
    """Calculate option price"""
    option_tree, _ = generate_tree(f, k, r, t, v, cp, n)
    return option_tree[0, 0]


def calculate_delta(
    f: float,
    k: float,
    r: float,
    t: float,
    v: float,
    cp: int,
    n: int = DEFAULT_STEP
) -> float:
    """Calculate option delta"""
    option_tree, underlying_tree = generate_tree(f, k, r, t, v, cp, n)

    option_price_change: float = option_tree[0, 1] - option_tree[1, 1]
    underlying_price_change: float = underlying_tree[0, 1] - underlying_tree[1, 1]

    delta: float = option_price_change / underlying_price_change
    return delta


def calculate_gamma(
    f: float,
    k: float,
    r: float,
    t: float,
    v: float,
    cp: int,
    n: int = DEFAULT_STEP
) -> float:
    """Calculate option gamma"""
    option_tree, underlying_tree = generate_tree(f, k, r, t, v, cp, n)

    gamma_delta_1: float = (option_tree[0, 2] - option_tree[1, 2]) / \
        (underlying_tree[0, 2] - underlying_tree[1, 2])
    gamma_delta_2: float = (option_tree[1, 2] - option_tree[2, 2]) / \
        (underlying_tree[1, 2] - underlying_tree[2, 2])

    gamma: float = (gamma_delta_1 - gamma_delta_2) / \
        (0.5 * (underlying_tree[0, 2] - underlying_tree[2, 2]))
    return gamma


def calculate_theta(
    f: float,
    k: float,
    r: float,
    t: float,
    v: float,
    cp: int,
    n: int = DEFAULT_STEP
) -> float:
    """Calcualte option theta"""
    option_tree, _ = generate_tree(f, k, r, t, v, cp, n)

    dt: float = t / n

    theta = (option_tree[1, 2] - option_tree[0, 0]) / (2 * dt)
    return theta


def calculate_vega(
    f: float,
    k: float,
    r: float,
    t: float,
    v: float,
    cp: int,
    n: int = DEFAULT_STEP
) -> float:
    """Calculate option vega"""
    price_1: float = calculate_price(f, k, r, t, v, cp, n)
    price_2: float = calculate_price(f, k, r, t, v * 1.001, cp, n)
    vega: float = (price_2 - price_1) / (v * 0.001)
    return vega


def calculate_greeks(
    f: float,
    k: float,
    r: float,
    t: float,
    v: float,
    cp: int,
    n: int = DEFAULT_STEP
) -> Tuple[float, float, float, float, float]:
    """Calculate option price and greeks"""
    dt: float = t / n
    option_tree, underlying_tree = generate_tree(f, k, r, t, v, cp, n)
    option_tree_vega, _ = generate_tree(f, k, r, t, v * 1.001, cp, n)

    # Price
    price: float = option_tree[0, 0]

    # Delta
    option_price_change: float = option_tree[0, 1] - option_tree[1, 1]
    underlying_price_change: float = underlying_tree[0, 1] - underlying_tree[1, 1]
    delta: float = option_price_change / underlying_price_change

    # Gamma
    gamma_delta_1: float = (option_tree[0, 2] - option_tree[1, 2]) / \
        (underlying_tree[0, 2] - underlying_tree[1, 2])
    gamma_delta_2: float = (option_tree[1, 2] - option_tree[2, 2]) / \
        (underlying_tree[1, 2] - underlying_tree[2, 2])
    gamma: float = (gamma_delta_1 - gamma_delta_2) / \
        (0.5 * (underlying_tree[0, 2] - underlying_tree[2, 2]))

    # Theta
    theta: float = (option_tree[1, 2] - option_tree[0, 0]) / (2 * dt)

    # Vega
    vega: float = (option_tree_vega[0, 0] - option_tree[0, 0]) / (0.001 * v)

    return price, delta, gamma, theta, vega


def calculate_impv(
    price: float,
    f: float,
    k: float,
    r: float,
    t: float,
    cp: int,
    n: int = DEFAULT_STEP
) -> float:
    """Calculate option implied volatility"""
    # Check option price must be position
    if price <= 0:
        return 0

    # Check if option price meets minimum value (exercise value)
    meet: bool = False

    if cp == 1 and price > (f - k):
        meet = True
    elif cp == -1 and price > (k - f):
        meet = True

    # If minimum value not met, return 0
    if not meet:
        return 0

    # Calculate implied volatility with Newton's method
    v: float = 0.3      # Initial guess of volatility

    for i in range(50):
        # Caculate option price and vega with current guess
        p: float = calculate_price(f, k, r, t, v, cp, n)
        vega: float = calculate_vega(f, k, r, t, v, cp, n)

        # Break loop if vega too close to 0
        if not vega:
            break

        # Calculate error value
        dx: float = (price - p) / vega

        # Check if error value meets requirement
        if abs(dx) < 0.00001:
            break

        # Calculate guessed implied volatility of next round
        v += dx

        # Check new volatility to be non-negative
        if v <= 0:
            return 0

    # Round to 4 decimal places
    v = round(v, 4)

    return v
