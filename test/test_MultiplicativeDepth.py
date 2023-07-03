import unittest
import numpy as np
from pyBFV.BFV_bigint import *


def generate_int(in_range: tuple, size=1):
    return np.array(np.random.randint(in_range[0], in_range[1], size=size, dtype=np.int64), dtype=object)


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Setting up the scheme
        # PARAMETERS MUST ADHERE TO THE FORMULA HERE:
        # https://bit-ml.github.io/blog/post/homomorphic-encryption-toy-implementation-in-python/
        self.desired_range = (2, 10)
        self.max_depth = 15
        # Scheme's params
        # polynomial modulus degree, SIZE parameter
        self.n = 2 ** 6
        # ciphertext modulus, MODULUS parameter
        self.q = 2 ** 500
        # plaintext modulus
        self.t = 2 ** 10

        # modulusswitching modulus
        self.p = self.q ** 3

        # polynomial modulus
        self.poly_mod = np.array([1] + [0] * (self.n - 1) + [1])

        # standard deviation for the error in the encryption, common value 3.2
        self.std1 = 1
        # standard deviation for the error in the evaluateKeyGen_v2
        self.std2 = 1
        # Base for decomposition
        self.BASE = 2

    def test_multiplicative_depth(self):
        """
        Selects two integers at random and continues multiplication until the result is incorrect, this can be used to
        estimate the multiplicative capabilities of the selected parameters
        :return:
        """
        pk, sk = keygen(self.n, self.q, self.poly_mod, self.std1)

        # EvaluateKeygen_version2
        rlk0, rlk1 = evaluate_keygen_v2(sk, self.n, self.q, self.poly_mod, self.p, self.std2)

        # PARTY 1 message
        m1 = int2base(127, 2)
        # PARTY 2 message
        m2 = int2base(33, 2)

        ct1 = encrypt(pk, self.n, self.q, self.t, self.poly_mod, m1, self.std1)
        ct2 = encrypt(pk, self.n, self.q, self.t, self.poly_mod, m2, self.std1)

        # Multiplication experiment
        e_mult = mul_cipher_v2(ct1, ct2, self.q, self.t, self.p, self.poly_mod, rlk0, rlk1)
        e_mult = mul_cipher_v2(e_mult, ct2, self.q, self.t, self.p, self.poly_mod, rlk0, rlk1)
        e_mult = mul_cipher_v2(e_mult, ct2, self.q, self.t, self.p, self.poly_mod, rlk0, rlk1)
        d_mult = decrypt(sk, self.q, self.t, self.poly_mod, e_mult)

        # Multiplication result
        print(base2int(d_mult, 2))

        # Iterate through random integer list
        rand_list = generate_int(self.desired_range, self.max_depth)
        expected_result = rand_list[0]
        m1 = int2base(expected_result, 2)
        ct1 = encrypt(pk, self.n, self.q, self.t, self.poly_mod, m1, self.std1)
        depth_reached = 1

        for num in rand_list[1:]:
            # Updating expected result in plaintext
            old_expected_r = expected_result
            expected_result = expected_result * num

            # Computing encoding and encryption for second term of the multiplication
            m2 = int2base(num, 2)
            ct2 = encrypt(pk, self.n, self.q, self.t, self.poly_mod, m2, self.std1)

            # Encrypted multiplication
            e_mult = mul_cipher_v2(ct1, ct2, self.q, self.t, self.p, self.poly_mod, rlk0, rlk1)
            # Reusing ct1 to contain multiplication result
            ct1 = e_mult

            # Decrypting and checking multiplication
            d_mult = decrypt(sk, self.q, self.t, self.poly_mod, e_mult)
            d_result = base2int(d_mult, 2)
            print(f"CURRENT DEPTH: {depth_reached}")
            print(f"EXPECTED RESULT: {old_expected_r}*{num}={expected_result}. ENCRYPTED RESULT: {d_result}")
            # Increase depth
            depth_reached += 1



if __name__ == '__main__':
    unittest.main()
