from random_variable import RandomVariable, gaussian_distribution

def test_var(var, n=1000):
    print(var)
    print(f"Expantacy(n={n}) = ", var.expantacy(n))
    print(f"Variance(n={n}) = ", var.variance(n))
    print(f"std(n={n}) = ", var.std(n))
    print()


if __name__ == "__main__":
    print("Constante variable")
    const = RandomVariable.constant_var(1,"1")
    test_var(const)

    print("Random variable with normal law")
    X = RandomVariable("X", gaussian_distribution(mu=-1.85, sigma=3.14))
    test_var(X)

    print("nuplet of the random variable above")
    X_n = X.nuplet(3)
    test_var(X_n)

    X_minus_X = X - X
    test_var(X_minus_X)

    X_plus_X = X + X
    test_var(X_plus_X)

    X_pow_2 = X**2
    test_var(X_pow_2)
