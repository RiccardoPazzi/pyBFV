# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from BFV import *
from CKKSEncoder import CKKSEncoder


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Scheme's parameters
    # polynomial modulus degree, SIZE parameter
    n = 2 ** 2
    # ciphertext modulus, MODULUS parameter
    q = 2 ** 14
    # plaintext modulus
    t = 8
    # base for relin_v1
    T = int(np.sqrt(q))
    # modulusswitching modulus
    p = q ** 3

    # polynomial modulus
    poly_mod = np.array([1] + [0] * (n - 1) + [1])
    print(poly_mod)

    # standard deviation for the error in the encryption, common value 3.2
    std1 = 1
    # standard deviation for the error in the evaluateKeyGen_v2
    std2 = 1

    # Multiparty procedures with 2 parties

    pk, sk = keygen(n, q, poly_mod, std1)

    # Generate keys for second party
    pk1, sk1 = party_key_gen(pk, n, q, poly_mod, std1)

    # Aggregate in a list
    pk_list = [pk, pk1]

    # Generate common public key
    common_pk = multi_common_key_gen(pk_list, n, q, poly_mod)
    common_sk = polyadd(sk, sk1, q, poly_mod)  # NOT TO BE USED IN PRODUCTION, ONLY FOR TESTING

    # Generate rlk0, rlk1
    # Generate Common Reference String
    common_a = gen_uniform_poly(n, q * p)
    # Each party generates their first tuple
    h0_0, h1_0, u_0 = compute_first_round_tuple(sk, common_a, n, std2, q, poly_mod, p)
    h0_1, h1_1, u_1 = compute_first_round_tuple(sk1, common_a, n, std2, q, poly_mod, p)

    # Sum the generated tuples into a single public pair
    h0, h1 = sum_first_round_tuples([h0_0, h0_1], [h1_0, h1_1], n, poly_mod, q * p)

    # Each party generates the second round tuples
    h0_prime_0, h1_prime_0 = compute_second_round_tuples(sk, u_0, h0, h1, p * q, poly_mod, n, std2)
    h0_prime_1, h1_prime_1 = compute_second_round_tuples(sk1, u_1, h0, h1, p * q, poly_mod, n, std2)

    # Generate the final relinearization keys
    rlk0, rlk1 = compute_final_eval_key([h0_prime_0, h0_prime_1], [h1_prime_0, h1_prime_1], h1, n, poly_mod, p * q)

    # Test to check if it works

    # PARTY 1 message
    m1 = int2base(4, 2)
    # PARTY 2 message
    m2 = int2base(2, 2)

    # Encryption
    e_m1 = encrypt(common_pk, n, q, t, poly_mod, m1, std1)
    e_m2 = encrypt(common_pk, n, q, t, poly_mod, m2, std1)

    # Summing the ctxt

    e_sum = add_cipher(e_m1, e_m2, q, poly_mod)

    # Decryption using sum of secret keys (NOT ACCESSIBLE)
    d_sum = decrypt(common_sk, q, t, poly_mod, e_sum)

    print(f"Common secret key: {common_sk}")
    common_sk_squared = poly.polymul(common_sk, common_sk) % q
    print(f"Common secret key squared: {common_sk_squared}")
    print(f"Secret key squared modulo poly_mod: {poly.polydiv(common_sk_squared, poly_mod)[1] % q}")
    print(d_sum)
    print("-----------------TEST DISTRIBUTED DECRYPTION ALGORITHM-------------")
    partial_ct = distributed_decryption_step(e_sum, sk, q, t, poly_mod)
    final_pt = distributed_decryption_step(partial_ct, sk1, q, t, poly_mod, last_step=True)
    print(final_pt)

    # Multiplication experiment
    e_mult = mul_cipher_v2(e_m1, e_m2, q, t, p, poly_mod, rlk0, rlk1)
    d_mult = decrypt(common_sk, q, t, poly_mod, e_mult)
    print(d_mult)
    print(polyadd_wm(rlk0, polymul_wm(rlk1, common_sk, poly_mod), poly_mod) % (p * q) // p)

    # Old relinearization procedure
    rlk0, rlk1 = evaluate_keygen_v2(common_sk, n, q, poly_mod, p, std2)
    e_mult = mul_cipher_v2(e_m1, e_m2, q, t, p, poly_mod, rlk0, rlk1)
    d_mult = decrypt(common_sk, q, t, poly_mod, e_mult)

    print(d_mult)
    print(polyadd_wm(rlk0, polymul_wm(rlk1, common_sk, poly_mod), poly_mod) % (p * q) // p)

