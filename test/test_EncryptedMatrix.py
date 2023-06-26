import unittest
import numpy as np
from pyBFV.BFV import *
from pyBFV.Ciphertext import EncryptedMatrix


def generate_int_matrix(values: tuple, random=False, shape=None):
    if random:
        return np.random.randint(values[0], values[1], shape, dtype=np.int64)
    matrix = [[1, 2], [3, 4]]
    return matrix


def generate_int(in_range: tuple):
    return np.random.randint(in_range[0], in_range[1], dtype=np.int64)


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Setting up the scheme parameters
        self.desired_shape = (20, 20)
        self.desired_range = (0, 100)
        # polynomial modulus degree, SIZE parameter
        # Limits the max number representable in base 2 (2 ^ n - 1)
        self.n = 2 ** 5
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
        self.t = 64
        # modulus switching modulus
        self.p = self.q/4
        # Polynomial modulus
        self.poly_mod = np.array([1] + [0] * (self.n - 1) + [1])
        # Std for encryption
        self.std1 = 1
        self.std2 = 1
        # Base for decomposition
        self.BASE = 2

    def test_decryption(self):
        # generate public and secret keys
        pk, sk = keygen(self.n, self.q, self.poly_mod, self.std1)

        params = (pk, self.n, self.q, self.t, self.p, self.poly_mod, self.std1)

        # Create test matrices
        int_matrix = generate_int_matrix(self.desired_range, True, self.desired_shape)

        c_1 = EncryptedMatrix()

        c_1.encrypts(int_matrix, params)
        print("CHECKING ORIGINAL MATRIX MATCHES DECRYPTED MATRIX...")
        decrypted_matrix = c_1.decrypt(sk)
        print(decrypted_matrix)
        # print("\n------------------------------------------------\n")
        self.assertTrue(np.array_equal(int_matrix, decrypted_matrix))  # add assertion here

    def test_sum(self):
        # generate public and secret keys
        pk, sk = keygen(self.n, self.q, self.poly_mod, self.std1)

        params = (pk, self.n, self.q, self.t, self.p, self.poly_mod, self.std1)

        # Create test matrices
        int_matrix = generate_int_matrix(self.desired_range, True, self.desired_shape)
        int_matrix2 = generate_int_matrix(self.desired_range, True, self.desired_shape)

        # Create Encrypted objects
        c_1 = EncryptedMatrix()
        c_2 = EncryptedMatrix()

        # Embed data
        c_1.encrypts(int_matrix, params)
        c_2.encrypts(int_matrix2, params)

        # Sum them up
        c_3 = c_1 + c_2
        decrypted_matrix = c_3.decrypt(sk)
        print("CHECKING SUM FUNCTION...")
        print(f"{int_matrix} + {int_matrix2} = {decrypted_matrix}")
        # print("\n---------------------------------\n")  # Visual separator
        self.assertTrue(np.array_equal(np.array(int_matrix) + np.array(int_matrix2), decrypted_matrix))

    def test_dot_product(self):
        # generate public, secret and relinearization keys
        pk, sk = keygen(self.n, self.q, self.poly_mod, self.std1)
        rlk = evaluate_keygen_v2(sk, self.n, self.q, self.poly_mod, self.p, self.std2)

        params = (pk, self.n, self.q, self.t, self.p, self.poly_mod, self.std1)

        # Create test matrices
        int_matrix = generate_int_matrix(self.desired_range, True, self.desired_shape)
        int_matrix2 = generate_int_matrix(self.desired_range, True, self.desired_shape)

        # Create Encrypted objects
        c_1 = EncryptedMatrix()
        c_2 = EncryptedMatrix()

        # Embed data
        c_1.encrypts(int_matrix, params)
        c_2.encrypts(int_matrix2, params)

        # Product computation
        c_4 = c_1.dot(c_2, rlk)
        decrypted_matrix = c_4.decrypt(sk)
        print("CHECKING DOT PRODUCT...")
        check_matrix = np.array(int_matrix).dot(np.array(int_matrix2))
        print(f"ERROR MATRIX = {decrypted_matrix - check_matrix}")
        self.assertTrue(np.array_equal(check_matrix, decrypted_matrix))

    def test_scale(self):
        # generate public, secret and relinearization keys
        pk, sk = keygen(self.n, self.q, self.poly_mod, self.std1)
        rlk = evaluate_keygen_v2(sk, self.n, self.q, self.poly_mod, self.p, self.std2)

        params = (pk, self.n, self.q, self.t, self.p, self.poly_mod, self.std1)

        # Create test matrices
        int_matrix = generate_int_matrix(self.desired_range, True, self.desired_shape)

        # Create Encrypted objects
        c_1 = EncryptedMatrix()
        scale = generate_int((1, 10))
        scale_poly = int2base(scale, self.BASE)
        e_scale = encrypt(pk, self.n, self.q, self.t, self.poly_mod, scale_poly, self.std1)

        # Embed data
        c_1.encrypts(int_matrix, params)

        # Scale the matrix
        c_6 = c_1.scale(e_scale, rlk)
        decrypted_matrix = c_6.decrypt(sk)
        print("CHECKING SCALE FUNCTION...")
        print(f"{scale} * {int_matrix} = {decrypted_matrix}")
        self.assertTrue(np.array_equal(scale * np.array(int_matrix), decrypted_matrix))


if __name__ == '__main__':
    unittest.main()
