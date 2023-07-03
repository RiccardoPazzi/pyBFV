import unittest
import numpy as np
from pyBFV.BFV import *


def generate_int(in_range: tuple):
    return np.random.randint(in_range[0], in_range[1], dtype=np.int64)


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Setting up the scheme
        # PARAMETERS MUST ADHERE TO THE FORMULA HERE:
        # https://bit-ml.github.io/blog/post/homomorphic-encryption-toy-implementation-in-python/
        self.desired_range = (2, 4)
        # polynomial modulus degree, SIZE parameter
        # Limits the max number representable in base 2 (2 ^ n - 1)
        self.n = 2 ** 3
        # 2 ** 6 doesn't work. Max parameters are 2 ** 5, q = 2^27, t = 64
        # print(f"BASE decomposition will work up to {2 ** self.n}")

        # ciphertext modulus, MODULUS parameter
        self.q = 2 ** 29

        # plaintext modulus
        # Some operations might still produce the expected results when the number exceeds t due to base decomposition
        # However t for sure limits the number of sums to t, because even with binary decomposition the 1's added will
        # eventually surpass e.g. 128
        # Limiting the number of sums to t will in turn limit the dot product which can only operate on tensors of the form
        # (.. x t) . (t x ..) where t is the dimension which will be eliminated from both tensors
        self.t = 4
        # modulus switching modulus
        self.p = self.q
        # Polynomial modulus
        self.poly_mod = np.array([1] + [0] * (self.n - 1) + [1])
        # Std for encryption
        self.std1 = 1
        self.std2 = 1
        # Base for decomposition
        self.BASE = 2

    def test_multiplicative_depth(self):
        """
        Selects two integers at random and continues multiplication until the result is incorrect, this can be used to
        estimate the multiplicative capabilities of the selected parameters
        :return:
        """
        # generate public, secret and relinearization keys
        pk, sk = keygen(self.n, self.q, self.poly_mod, self.std1)
        rlk = evaluate_keygen_v2(sk, self.n, self.q, self.poly_mod, self.p, self.std2)

        params = (pk, self.n, self.q, self.t, self.p, self.poly_mod, self.std1)
        correct_result = True
        mul_depth = 1
        expected_result = generate_int(self.desired_range)
        e_m1 = int2base(expected_result, self.BASE)
        ctx1 = encrypt(pk, self.n, self.q, self.t, self.poly_mod, e_m1, self.std1)
        while correct_result:
            m1 = expected_result
            m2 = generate_int(self.desired_range)
            expected_result = m1 * m2
            # PARTY 2 message
            e_m2 = int2base(m2, self.BASE)
            ctx2 = encrypt(pk, self.n, self.q, self.t, self.poly_mod, e_m2, self.std1)
            # Multiplication experiment
            ctx1 = mul_cipher_v2(ctx1, ctx2, self.q, self.t, self.p, self.poly_mod, rlk[0], rlk[1])
            d_mult = decrypt(sk, self.q, self.t, self.poly_mod, ctx1)
            # Decode result
            actual_result = base2int(d_mult, self.BASE)
            print(f"MULTIPLICATIVE DEPTH: {mul_depth}")
            print(f"Expected result: {expected_result}, actual result: {actual_result}")
            if expected_result > (2 ** self.n):
                print(f"Result is over the bit capacity!")
            self.assertTrue(actual_result == expected_result)
            mul_depth += 1

    def test_addition_depth(self):
        """
        Selects two integers at random and continues adding until the result is incorrect, this can be used to
        estimate the additive capabilities of the selected parameters
        :return:
        """
        # generate public and secret keys
        pk, sk = keygen(self.n, self.q, self.poly_mod, self.std1)

        params = (pk, self.n, self.q, self.t, self.p, self.poly_mod, self.std1)
        correct_result = False
        sum_depth = 1
        expected_result = generate_int(self.desired_range)
        while correct_result:
            m1 = expected_result
            m2 = generate_int(self.desired_range)
            expected_result = m1 + m2
            # PARTY 1 message
            e_m1 = int2base(m1, self.BASE)
            # PARTY 2 message
            e_m2 = int2base(m2, self.BASE)
            # Encryption
            ctx1 = encrypt(pk, self.n, self.q, self.t, self.poly_mod, e_m1, self.std1)
            ctx2 = encrypt(pk, self.n, self.q, self.t, self.poly_mod, e_m2, self.std1)
            # Multiplication experiment
            e_sum = add_cipher(ctx1, ctx2, self.q, self.poly_mod)
            d_sum = decrypt(sk, self.q, self.t, self.poly_mod, e_sum)
            # Decode result
            actual_result = base2int(d_sum, self.BASE)
            if sum_depth % 50 == 0:
                print(f"MULTIPLICATIVE DEPTH: {sum_depth}")
                print(f"Expected result: {expected_result}, actual result: {actual_result}")
                if expected_result > (2 ** self.n):
                    print(f"Result is over the bit capacity!")
            self.assertTrue(actual_result == expected_result)
            sum_depth += 1


if __name__ == '__main__':
    unittest.main()
