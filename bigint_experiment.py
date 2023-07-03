from pyBFV.BFV_bigint import *


if __name__ == '__main__':
    # Scheme's params
    # polynomial modulus degree, SIZE parameter
    n = 2 ** 5
    # ciphertext modulus, MODULUS parameter
    q = 2 ** 60
    # plaintext modulus
    t = 2 ** 3

    # modulusswitching modulus
    p = q ** 3

    # polynomial modulus
    poly_mod = np.array([1] + [0] * (n - 1) + [1])

    # standard deviation for the error in the encryption, common value 3.2
    std1 = 1
    # standard deviation for the error in the evaluateKeyGen_v2
    std2 = 1

    # Multiparty procedures with 2 parties

    pk, sk = keygen(n, q, poly_mod, std1)

    # EvaluateKeygen_version2
    rlk0, rlk1 = evaluate_keygen_v2(sk, n, q, poly_mod, p, std2)
    print(pk, sk)
    print(rlk0, rlk1)

    # PARTY 1 message
    m1 = int2base(127, 2)
    # PARTY 2 message
    m2 = int2base(33, 2)
    print(m1, m2)

    ct1 = encrypt(pk, n, q, t, poly_mod, m1, std1)
    ct2 = encrypt(pk, n, q, t, poly_mod, m2, std1)

    print(ct1, ct2)
    print(decrypt(sk, q, t, poly_mod, ct1))
    print(decrypt(sk, q, t, poly_mod, ct2))

    # Multiplication experiment
    e_mult = mul_cipher_v2(ct1, ct2, q, t, p, poly_mod, rlk0, rlk1)
    e_mult = mul_cipher_v2(e_mult, ct2, q, t, p, poly_mod, rlk0, rlk1)
    e_mult = mul_cipher_v2(e_mult, ct2, q, t, p, poly_mod, rlk0, rlk1)
    d_mult = decrypt(sk, q, t, poly_mod, e_mult)
    e_sum = add_cipher(ct1, ct2, q, poly_mod)
    e_sum = add_cipher(e_sum, ct1, q, poly_mod)
    e_sum = add_cipher(e_sum, ct1, q, poly_mod)
    d_sum = decrypt(sk, q, t, poly_mod, e_sum)

    # Multiplication result
    print(base2int(d_mult, 2))
    print(d_sum)
