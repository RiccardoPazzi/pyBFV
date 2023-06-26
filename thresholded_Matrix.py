from pyBFV.BFV import *
from pyBFV.Ciphertext import EncryptedMatrix

# PARAMETER SELECTION
# polynomial modulus degree, SIZE parameter
n = 2 ** 4
# ciphertext modulus, MODULUS parameter
q = 2 ** 30
# plaintext modulus
t = 8
# base for relin_v1
T = int(np.sqrt(q))
# modulusswitching modulus
p = q

# polynomial modulus
poly_mod = np.array([1] + [0] * (n - 1) + [1])
# standard deviation for the error in the encryption, common value 3.2
std1 = 1
# standard deviation for the error in the evaluateKeyGen_v2
std2 = 1


def TwoPartyKeyGen():
    # Multiparty procedures with 2 parties, returns the

    # Aggregate in a list the public keys
    pk_list = [pk, pk1]

    # Generate common public key
    common_pk = multi_common_key_gen(pk_list, n, q, poly_mod)
    # common_sk = polyadd(sk, sk1, q, poly_mod)  # NOT TO BE USED IN PRODUCTION, ONLY FOR TESTING

    # Generate rlk0, rlk1
    # Generate Common Reference String GLoBAL
    common_a = gen_uniform_poly(n, q * p)
    # Each party generates their first tuple LOCAL
    h0_0, h1_0, u_0 = compute_first_round_tuple(sk, common_a, n, std2, q, poly_mod, p)
    h0_1, h1_1, u_1 = compute_first_round_tuple(sk1, common_a, n, std2, q, poly_mod, p)

    # Sum the generated tuples into a single public pair GLOBAL
    h0, h1 = sum_first_round_tuples([h0_0, h0_1], [h1_0, h1_1], n, poly_mod, q * p)

    # Each party generates the second round tuples LOCAL
    h0_prime_0, h1_prime_0 = compute_second_round_tuples(sk, u_0, h0, h1, p * q, poly_mod, n, std2)
    h0_prime_1, h1_prime_1 = compute_second_round_tuples(sk1, u_1, h0, h1, p * q, poly_mod, n, std2)

    # Generate the final relinearization keys
    rlk0, rlk1 = compute_final_eval_key([h0_prime_0, h0_prime_1], [h1_prime_0, h1_prime_1], h1, n, poly_mod, p * q)
    return common_pk, (rlk0, rlk1)


if __name__ == '__main__':
    # LOCAL KEY GENERATION

    # Public and private keys are generated here to access them from wherever
    # In the future a Party class will be created to handle the keys locally
    pk, sk = keygen(n, q, poly_mod, std1)

    # Generate keys for second party
    pk1, sk1 = party_key_gen(pk, n, q, poly_mod, std1)

    common_pk, rlk = TwoPartyKeyGen()
    common_sk = polyadd(sk, sk1, q, poly_mod)  # NOT TO BE USED IN PRODUCTION, ONLY FOR TESTING

    # Generate parameters tuple
    params = (common_pk, n, q, t, p, poly_mod, std1)

    # PARTY 1 matrix
    int_matrix = [[1, 2], [3, 4]]
    # PARTY 2 matrix
    int_matrix2 = [[1, 2], [3, 4]]

    # Create Encrypted objects
    c_1 = EncryptedMatrix()
    c_2 = EncryptedMatrix()

    # Embed data
    c_1.encrypts(int_matrix, params)
    c_2.encrypts(int_matrix2, params)

    # Product computation
    c_4 = c_1.dot(c_2, rlk)
    decrypted_matrix = c_4.decrypt(common_sk)
    print(decrypted_matrix)
    # Another product
    c_5 = c_1.dot(c_4, rlk)
    decrypted_matrix = c_5.decrypt(common_sk)
    print(decrypted_matrix)
    # Another product
    c_5 = c_1.dot(c_5, rlk)
    decrypted_matrix = c_5.decrypt(common_sk)
    print(decrypted_matrix)
    dec_matrix = c_4.distributed_decrypt(sk)
    final_result = dec_matrix.distributed_decrypt(sk1, True)
    print(final_result)

