from vnpy_optionmaster.pricing import (
    binomial_tree as binomial_tree_python,
    black_76 as black_76_python,
    black_scholes as black_scholes_python,
)

from vnpy_optionmaster.pricing import (
    binomial_tree_cython,
    black_76_cython,
    black_scholes_cython,
)


s = 100
k = 100
r = 0.03
t = 0.1
v = 0.2
cp = 1


for py_model, cy_model in [
    (binomial_tree_python, binomial_tree_cython),
    (black_76_python, black_76_cython),
    (black_scholes_python, black_scholes_cython)
]:
    print("-" * 30)

    for model in [py_model, cy_model]:
        print(" ")
        print(model.__name__)
        print("price", model.calculate_price(s, k, r, t, v, cp))
        print("delta", model.calculate_delta(s, k, r, t, v, cp))
        print("gamma", model.calculate_gamma(s, k, r, t, v, cp))
        print("theta", model.calculate_theta(s, k, r, t, v, cp))
        print("vega", model.calculate_vega(s, k, r, t, v, cp))
        print("greeks", model.calculate_greeks(s, k, r, t, v, cp))
