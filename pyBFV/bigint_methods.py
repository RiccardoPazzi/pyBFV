import numpy as np
import numpy.polynomial.polyutils as pu


def normalize(poly):
    while len(poly) > 0 and poly[-1] == 0:
        poly.pop()
    if len(poly) == 0:
        poly.append(0)


# ------Helper functions needed to circumvent numpy methods which are subject to overflow
def roundintdiv(num_array, div):
    threshold = div // 2
    div_even = div % 2
    result_array = [0] * len(num_array)
    for idx, number in enumerate(num_array):
        result = number // div
        rem = number % div
        if rem > threshold or (rem == threshold and div_even == 0):
            result_array[idx] = result + 1
        else:
            result_array[idx] = result
    return np.array(result_array, dtype=object)


def polymul(x, y):
    """
    Multiply two big integer polynomials, this method must be used in case the numpy limit of 64 bits is exceeded,
    otherwise numpy will produce wrong results
    :param x:
    :param y:
    :return:
    """
    m = len(x)
    n = len(y)
    result = [0] * (m + n - 1)
    for i in range(m):
        for j in range(n):
            result[i + j] += x[i] * y[j]
    return np.array(result, dtype=object)


def polydiv(c1, c2):
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError()

    # note: this is more efficient than `pu._div(polymul, c1, c2)`
    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return c1[:1]*0, c1
    elif lc2 == 1:
        return c1//c2[-1], c1[:1]*0
    else:
        dlen = lc1 - lc2
        scl = c2[-1]
        c2 = c2[:-1]//scl
        i = dlen
        j = lc1 - 1
        while i >= 0:
            c1[i:j] -= c2*c1[j]
            i -= 1
            j -= 1
        return c1[j+1:]//scl, np.array(pu.trimseq(c1[:j+1]), dtype=object)


def polyadd(c1, c2):
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if len(c1) > len(c2):
        c1[:c2.size] += c2
        ret = c1
    else:
        c2[:c1.size] += c1
        ret = c2
    return pu.trimseq(ret)


if __name__ == "__main__":
    # Test poly division
    first_poly = np.array([105000000000000000000000, 10, 20, 30, 0, 0, 0], dtype=object)
    second_poly = np.array([1, 0, 0, 1], dtype=object)
    print(polydiv(first_poly, second_poly))
