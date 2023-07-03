"""A basic homomorphic encryption scheme inspired from [FV12] https://eprint.iacr.org/2012/144.pdf
The starting point of our Python implementation is this github gist: https://gist.github.com/youben11/f00bc95c5dde5e11218f14f7110ad289.
Disclaimer: Our toy implementation is not meant to be secure or optimized for efficiency.
We did it to better understand the inner workings of the [FV12] scheme, so you can use it as a learning tool.
"""

import numpy as np
import random
from numpy.polynomial import polynomial as poly
import pyBFV.bigint_methods as bigint


# TODO: Create a class to contain all the methods and CryptoParameters as fields
# ------Functions for polynomial evaluations mod poly_mod only------
def polymul_wm(x, y, poly_mod):
    """Multiply two polynomials
    Args:
        x, y: two polynomials to be multiplied.
        poly_mod: polynomial modulus.
    Returns:
        A polynomial in Z[X]/(poly_mod).
    """
    return bigint.polydiv(bigint.polymul(x, y), poly_mod)[1]


def polyadd_wm(x, y, poly_mod):
    """Add two polynomials
        Args:
            x, y: two polynomials to be added.
            poly_mod: polynomial modulus.
        Returns:
            A polynomial in Z[X]/(poly_mod).
        """
    return bigint.polydiv(poly.polyadd(x, y), poly_mod)[1]


# ==============================================================

# ------Functions for polynomial evaluations both mod poly_mod and mod q-----
def polymul(x, y, modulus, poly_mod):
    """Multiply two polynomials
    Args:
        x, y: two polynomials to be multiplied.
        modulus: coefficient modulus.
        poly_mod: polynomial modulus.
    Returns:
        A polynomial in Z_modulus[X]/(poly_mod).
    """
    return bigint.polydiv(bigint.polymul(x, y) %
                          modulus, poly_mod)[1] % modulus


def polyadd(x, y, modulus, poly_mod):
    """Add two polynomials
    Args:
        x, y: two polynoms to be added.
        modulus: coefficient modulus.
        poly_mod: polynomial modulus.
    Returns:
        A polynomial in Z_modulus[X]/(poly_mod).
    """
    return bigint.polydiv(poly.polyadd(x, y) %
                          modulus, poly_mod)[1] % modulus


# ==============================================================

# -------Functions for random polynomial generation--------
def gen_binary_poly(size):
    """Generates a polynomial with coefficients in {-1, 0, 1}
    Args:
        size: number of coefficients, size-1 being the degree of the
            polynomial.
    Returns:
        array of coefficients with the coeff[i] being
        the coeff of x ^ i.
    """
    return np.random.randint(-1, 2, size, dtype=np.int64)


def gen_uniform_poly(size, modulus):
    """Generates a polynomial with coeffecients being integers in Z_modulus
    Args:
        size: number of coeffcients, size-1 being the degree of the
            polynomial.
    Returns:
        array of coefficients with the coeff[i] being
        the coeff of x ^ i.
    """
    return np.array([random.randint(0, modulus-1) for _ in range(size)], dtype=object)


def gen_normal_poly(size, mean, std):
    """Generates a polynomial with coefficients in a normal distribution
    of mean=mean and a standard deviation std, then discretize it.
    Args:
        size: number of coefficients, size-1 being the degree of the
            polynomial.
    Returns:
        array of coefficients with the coeff[i] being
        the coeff of x ^ i.
    """
    return np.int64(np.random.normal(mean, std, size=size))


# ==============================================================

# -------- Function for returning n's coefficients in base b ( lsb is on the left) ---
def int2base(n, b):
    """Generates the base decomposition of an integer n.
    Args:
        n: integer to be decomposed.
        b: base.
    Returns:
        array of coefficients from the base decomposition of n
        with the coeff[i] being the coeff of b ^ i.
    """
    if n < b:
        return [n]
    else:
        return [n % b] + int2base(n // b, b)


def base2int(base_vector, b):
    result = 0
    for idx, value in enumerate(base_vector):
        result += value * (b ** idx)
    return result

    # ------ Functions for keygen, encryption and decryption ------


def keygen(size, modulus, poly_mod, std1):
    """Generate a public and secret keys
    Args:
        size: size of the polynoms for the public and secret keys.
        modulus: coefficient modulus.
        poly_mod: polynomial modulus.
        std1: standard deviation of the error.
    Returns:
        Public and secret key.
    """
    s = gen_binary_poly(size)
    a = gen_uniform_poly(size, modulus)
    e = gen_normal_poly(size, 0, std1)
    b = polyadd(polymul(-a, s, modulus, poly_mod), -e, modulus, poly_mod)
    return (b, a), s


def evaluate_keygen_v2(sk, size, modulus, poly_mod, extra_modulus, std2):
    """Generate a relinearization key using version 2.
        Args:
            sk: secret key.
            size: size of the polynomials.
            modulus: coefficient modulus.
            poly_mod: polynomial modulus.
            extra_modulus: the "p" modulus for modulus switching.
            std2: standard deviation for the error distribution.
        Returns:
            rlk0, rlk1: relinearization key.
        """
    new_modulus = modulus * extra_modulus
    a = gen_uniform_poly(size, new_modulus)
    e = gen_normal_poly(size, 0, std2)
    secret_part = extra_modulus * bigint.polymul(sk, sk)

    b = polyadd_wm(
        polymul_wm(-a, sk, poly_mod),
        polyadd_wm(-e, secret_part, poly_mod), poly_mod) % new_modulus
    # b = -a * sk - e + p * sk^2
    return b, a


def encrypt(pk, size, q, t, poly_mod, m, std1):
    """Encrypt an integer.
    Args:
        pk: public-key.
        size: size of polynomials.
        q: ciphertext modulus.
        t: plaintext modulus.
        poly_mod: polynomial modulus.
        m: plaintext message, as an integer vector (of length <= size) with entries mod t.
    Returns:
        Tuple representing a ciphertext.
    """
    m = np.array(m + [0] * (size - len(m)), dtype=object) % t
    delta = q // t
    scaled_m = delta * m
    e1 = gen_normal_poly(size, 0, std1)
    e2 = gen_normal_poly(size, 0, std1)
    u = gen_binary_poly(size)
    ct0 = polyadd(
        polyadd(
            polymul(pk[0], u, q, poly_mod),
            e1, q, poly_mod),
        scaled_m, q, poly_mod
    )
    ct1 = polyadd(
        polymul(pk[1], u, q, poly_mod),
        e2, q, poly_mod
    )
    return ct0, ct1


def decrypt(sk, q, t, poly_mod, ct):
    """Decrypt a ciphertext.
    Args:
        sk: secret-key.
        size: size of polynomials.
        q: ciphertext modulus.
        t: plaintext modulus.
        poly_mod: polynomial modulus.
        ct: ciphertext.
    Returns:
        Integer vector representing the plaintext.
    """
    scaled_pt = polyadd(
        polymul(ct[1], sk, q, poly_mod),
        ct[0], q, poly_mod
    )

    decrypted_poly = bigint.roundintdiv(t * scaled_pt, q) % t
    decryption_debug = (t * scaled_pt / q) % t
    return decrypted_poly


# ==============================================================


# ------Function for adding and multiplying encrypted values------
def add_plain(ct, pt, q, t, poly_mod):
    """Add a ciphertext and a plaintext.
    Args:
        ct: ciphertext.
        pt: integer to add.
        q: ciphertext modulus.
        t: plaintext modulus.
        poly_mod: polynomial modulus.
    Returns:
        Tuple representing a ciphertext.
    """
    size = len(poly_mod) - 1
    # encode the integer into a plaintext polynomial
    m = np.array(pt + [0] * (size - len(pt)), dtype=object) % t
    delta = q // t
    scaled_m = delta * m
    new_ct0 = polyadd(ct[0], scaled_m, q, poly_mod)
    return (new_ct0, ct[1])


def add_cipher(ct1, ct2, q, poly_mod):
    """Add a ciphertext and a ciphertext.
    Args:
        ct1, ct2: ciphertexts.
        q: ciphertext modulus.
        poly_mod: polynomial modulus.
    Returns:
        Tuple representing a ciphertext.
    """
    new_ct0 = polyadd(ct1[0], ct2[0], q, poly_mod)
    new_ct1 = polyadd(ct1[1], ct2[1], q, poly_mod)
    return (new_ct0, new_ct1)


def mul_plain(ct, pt, q, t, poly_mod):
    """Multiply a ciphertext and a plaintext.
    Args:
        ct: ciphertext.
        pt: integer polynomial to multiply.
        q: ciphertext modulus.
        t: plaintext modulus.
        poly_mod: polynomial modulus.
    Returns:
        Tuple representing a ciphertext.
    """
    size = len(poly_mod) - 1
    # encode the integer polynomial into a plaintext vector of size=size
    m = np.array(pt + [0] * (size - len(pt)), dtype=object) % t
    new_c0 = polymul(ct[0], m, q, poly_mod)
    new_c1 = polymul(ct[1], m, q, poly_mod)
    return (new_c0, new_c1)


def multiplication_coeffs(ct1, ct2, q, t, poly_mod):
    """Multiply two ciphertexts.
        Args:
            ct1: first ciphertext.
            ct2: second ciphertext
            q: ciphertext modulus.
            t: plaintext modulus.
            poly_mod: polynomial modulus.
        Returns:
            Triplet (c0,c1,c2) encoding the multiplied ciphertexts.
        """

    c_0 = bigint.roundintdiv(polymul_wm(ct1[0], ct2[0], poly_mod) * t, q) % q
    c_1 = bigint.roundintdiv(
        polyadd_wm(polymul_wm(ct1[0], ct2[1], poly_mod), polymul_wm(ct1[1], ct2[0], poly_mod), poly_mod) * t, q) % q
    c_2 = bigint.roundintdiv(polymul_wm(ct1[1], ct2[1], poly_mod) * t, q) % q
    return c_0, c_1, c_2


def mul_cipher_v2(ct1, ct2, q, t, p, poly_mod, rlk0, rlk1):
    """Multiply two ciphertexts.
    Args:
        ct1: first ciphertext.
        ct2: second ciphertext.
        q: ciphertext modulus.
        t: plaintext modulus.
        p: modulus-swithcing modulus.
        poly_mod: polynomial modulus.
        rlk0: output of the EvaluateKeygen_v2 function.
        rlk1: output of the EvaluateKeygen_v2 function.
    Returns:
        Tuple representing a ciphertext.
    """
    c_0, c_1, c_2 = multiplication_coeffs(ct1, ct2, q, t, poly_mod)

    c_20 = bigint.roundintdiv(polymul_wm(c_2, rlk0, poly_mod), p) % q
    c_21 = bigint.roundintdiv(polymul_wm(c_2, rlk1, poly_mod), p) % q

    new_c0 = polyadd_wm(c_0, c_20, poly_mod) % q
    new_c1 = polyadd_wm(c_1, c_21, poly_mod) % q
    return new_c0, new_c1


# ==============================================================
# /                                                            /
# /                    MULTIPARTY OPERATIONS                   /
# /                        Key Generation                      /
# ==============================================================

def party_key_gen(pk, size, modulus, poly_mod, std1):
    """
        Generates the keys of parties after the CRS is generated (first element of the pk)
        :param: pk -> public key of anyone in the party
        :return: party_pk, party_sk
    """
    a = pk[1]
    s = gen_binary_poly(size)
    e = gen_normal_poly(size, 0, std1)
    b = polyadd(polymul(-a, s, modulus, poly_mod), -e, modulus, poly_mod)
    party_pk = (b, a)
    return party_pk, s


def multi_common_key_gen(pk_list, size_of_poly, modulus, poly_mod):
    """
    Generates the common public key starting from a list of public keys
    :param: pk_list -> public key list from which to generate the common key
    :return: common_pk
    """
    list_of_b = [x for x, y in pk_list]  # Take the second element (b) for all public keys
    sum_of_p = np.zeros(size_of_poly)

    # Sum all the second parts of the public keys
    for p in list_of_b:
        sum_of_p = polyadd(sum_of_p, p, modulus, poly_mod)

    # Create common public key
    common_pk = (sum_of_p, pk_list[0][1])
    return common_pk


# ==============================================================
# /                                                            /
# /                    MULTIPARTY OPERATIONS                   /
# /                       Relinearization                      /
# ==============================================================
"""
A single function cannot be employed since parties possess only their local secret share, not the entire secret key.
Therefore multiple rounds are needed, each party calls a function, then shares with the others
the result of the computation. This procedure follows the BFV threshold relinearization procedure from:
Christian Mouchet, Juan Troncoso-Pastoriza, Jean-Philippe Bossuat, and Jean-Pierre Hubaux: Multiparty Homomorphic 
Encryption from Ring-Learning-with-Errors
The weight decomposition procedure is not employed for clarity, however this means the coefficients will be much bigger
A similar approach for single-key HE is presented in the original FV scheme paper:
Somewhat Practical Fully Homomorphic Encryption : Junfeng Fan and Frederik Vercauteren
"""


def compute_first_round_tuple(sk, a, size, std, modulus, poly_mod, extra_modulus):
    """
    Generates the first tuple which has to be shared for all parties and then will be aggregated
    :param extra_modulus:
    :param a: The CRS (common polynomial) to all parties, has to be of matching size and modulus
    :param poly_mod: The cyclotomic polinomyal X^N + 1 used as modulus
    :param modulus: The parameter q, modulus of the coefficient in the polynomial
    :param std:
    :param mean:
    :param size:
    :param sk: Secret share of the party
    :return: (h0, h1, u) tuple (h0, h1) created by party i which will be shared and ui which is an additional secret
    polynomial
        """
    u = gen_binary_poly(size)
    e0 = gen_normal_poly(size, mean=0, std=std)
    e1 = gen_normal_poly(size, mean=0, std=std)
    new_modulus = modulus * extra_modulus
    scaled_sk = extra_modulus * sk
    # a = gen_uniform_poly(size, new_modulus)
    h0 = polyadd_wm(polymul_wm(-u, a, poly_mod), polyadd_wm(scaled_sk, e0, poly_mod), poly_mod) % new_modulus
    h1 = polyadd_wm(polymul_wm(sk, a, poly_mod), e1, poly_mod) % new_modulus
    # h0 = -u * a + p * sk + e0
    # h1 = sk * a + e1
    return h0, h1, u


def sum_first_round_tuples(list_h0, list_h1, size, poly_mod, new_modulus):
    """
    This function would be called by a central server which collects all the fragments of h0,h1 or by
    all parties after having shared their respective h0,h1 tuple
    :param new_modulus:
    :param poly_mod:
    :param size:
    :param list_h0: list of h0 computed by all parties
    :param list_h1: list of h1 computed by all parties
    :return: (h0_sum, h1_sum)
    """
    sum_h0 = np.zeros(size)
    sum_h1 = np.zeros(size)
    for h0 in list_h0:
        sum_h0 = polyadd_wm(sum_h0, h0, poly_mod) % new_modulus
    for h1 in list_h1:
        sum_h1 = polyadd_wm(sum_h1, h1, poly_mod) % new_modulus
    return sum_h0, sum_h1


def compute_second_round_tuples(sk, u_i, h0sum, h1sum, new_modulus, poly_mod, size, std):
    """
    Computes the second round of fragments which need to be shared among parties or sent to a common server
    :param sk:
    :param u_i:
    :param h0sum:
    :param h1sum:
    :param new_modulus:
    :param poly_mod:
    :param size:
    :param std:
    :return:
    """
    # Sample two error polynomials
    e2 = gen_normal_poly(size, mean=0, std=std)
    e3 = gen_normal_poly(size, mean=0, std=std)
    # Compute h0_prime and h1_prime
    h0_prime = polyadd_wm(polymul_wm(sk, h0sum, poly_mod), e2, poly_mod) % new_modulus
    h1_prime = polyadd_wm(polymul_wm(polyadd_wm(u_i, -sk, poly_mod), h1sum, poly_mod), e3, poly_mod) % new_modulus
    return h0_prime, h1_prime


def compute_final_eval_key(h0_prime_list, h1_prime_list, h1_sum, size, poly_mod, new_modulus):
    """
    Computes the common Eval key from the shares h0_prime and h1_prime and the public h1_sum
    :param h0_prime_list:
    :param h1_prime_list:
    :param h1_sum:
    :param size:
    :param poly_mod:
    :param new_modulus:
    :return:
    """
    sum_h0_p = np.zeros(size)
    sum_h1_p = np.zeros(size)
    for h0 in h0_prime_list:
        sum_h0_p = polyadd_wm(sum_h0_p, h0, poly_mod) % new_modulus
    for h1 in h1_prime_list:
        sum_h1_p = polyadd_wm(sum_h1_p, h1, poly_mod) % new_modulus
    # The evaluation key result is sum(h0') + sum(h1'), h1
    sum_h0_h1 = polyadd_wm(sum_h0_p, sum_h1_p, poly_mod) % new_modulus
    return np.array(sum_h0_h1, dtype=object), np.array(h1_sum, dtype=object)


# ==============================================================
# /                                                            /
# /                    MULTIPARTY OPERATIONS                   /
# /                          Decryption                        /
# ==============================================================
def distributed_decryption_step(ctxt, sk, q, t, poly_mod, last_step=False):
    """
    A single step of the distributed decryption procedure, this step
    must be repeated for all the parties (therefore with all the secret keys)
    :param ctxt:
    :param sk:
    :param q:
    :param t:
    :param poly_mod:
    :param last_step: Set to True if it's the last party step, will return the
    decrypted result
    :return: partial decryption if last_step=False, final decryption otherwise
    """
    new_ctxt = (polyadd(
        polymul(ctxt[1], sk, q, poly_mod),
        ctxt[0], q, poly_mod
    ), ctxt[1])
    if last_step:
        # De-scale the message and return the result
        decrypted_poly = bigint.roundintdiv(t * new_ctxt[0], q) % t
        return decrypted_poly
    else:
        # Return partial result, the message is still scaled
        return new_ctxt
