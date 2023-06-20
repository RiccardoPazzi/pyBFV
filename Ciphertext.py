"""
This is a utility class that allows to use the standard multiplication (*) and sum (+) operations on encrypted
ciphertext
"""
from BFV import *
import numpy as np

BASE = 2


class EncryptedMatrix:
    def __init__(self, from_array=None, parameters=None):
        if from_array is None:
            self.initialized = False # When false the matrix can encrypt an integer vector
            return
        self.pk, self.size, self.q, self.t, self.poly_mod, self.std1 = parameters
        self.packed_parameters = parameters
        self.shape = from_array.shape
        self.poly_matrix = from_array
        self.initialized = True

    def encrypts(self, int_vector, parameters: tuple):
        """
        Takes as input a matrix of integers and converts it into the corresponding polynomial representation
        using binary base encoding
        """
        if self.initialized:
            print("WARNING! Overriding an encrypted array which was already stored into this Object")
        self.pk, self.size, self.q, self.t, self.poly_mod, self.std1 = parameters
        self.packed_parameters = parameters
        np_input = np.array(int_vector, dtype=np.int64)
        self.shape = np_input.shape + (2, self.size)
        self.poly_matrix = np.zeros(self.shape)
        self.initialized = True
        # The matrix must contain two polynomials of degree size - 1, so we create a zero matrix with the correct shape

        for idx, integer in np.ndenumerate(np_input):
            m = int2base(integer, BASE)  # Convert into binary representation
            # Since the LSB is on the left
            # Create pair of polynomials corresponding to index
            ct0, ct1 = encrypt(self.pk, self.size, self.q, self.t, self.poly_mod, m, self.std1)
            # We save the two polynomials inside the correct location of the matrix
            # They might be shorter than the required dimension size, therefore we pad with zeros
            ct0 = np.pad(ct0, (0, self.size - len(ct0)), constant_values=0)
            ct1 = np.pad(ct1, (0, self.size - len(ct1)), constant_values=0)
            if len(ct0) < self.size:
                b = 1
            self.poly_matrix[idx + (0,)] += ct0
            self.poly_matrix[idx + (1,)] += ct1

        print(self.poly_matrix)

    def __mul__(self, other):
        # We iterate through the entire matrix except the last two dimensions which store the polynomials
        # The shape will be calculated as usual for the matrix product A x B = C where A.shape = [3,4] B.shape = [4,1]
        # Therefore C.shape will be [3,1]
        return self

    def __add__(self, other):
        # In order to add two matrices they must have the same shape
        assert self.shape == other.shape, "Shapes do not match!"
        poly_shape = self.shape[:-2]
        result = np.zeros(self.shape)
        for idx in np.ndindex(poly_shape):
            # Extract ciphertexts for the two matrices
            ct0 = self.poly_matrix[idx + (0,)]
            ct1 = self.poly_matrix[idx + (1,)]
            ct0_other = other.poly_matrix[idx + (0,)]
            ct1_other = other.poly_matrix[idx + (1,)]
            # Sum the two ciphertexts
            sum_ct0, sum_ct1 = add_cipher((ct0, ct1), (ct0_other, ct1_other), self.q, self.poly_mod)
            # Add ciphertexts to result, once more we need to pad with zeroes
            sum_ct0 = np.pad(sum_ct0, (0, self.size - len(sum_ct0)), constant_values=0)
            sum_ct1 = np.pad(sum_ct1, (0, self.size - len(sum_ct1)), constant_values=0)
            result[idx + (0,)] += sum_ct0
            result[idx + (1,)] += sum_ct1
        # Convert result into EncryptedMatrix
        e_result = EncryptedMatrix(result, self.packed_parameters)
        return e_result

    def decrypt(self, secret_key):
        out_shape = self.shape[:-2]
        out_matrix = np.zeros(out_shape, dtype=np.int64)
        for idx in np.ndindex(out_shape):
            ct0 = self.poly_matrix[idx + (0,)]
            ct1 = self.poly_matrix[idx + (1,)]
            # Generate the decrypted polynomial, which has to be converted into an integer
            decrypted_poly = decrypt(secret_key, self.q, self.t, self.poly_mod, (ct0, ct1))

            out_matrix[idx] += base2int(decrypted_poly, BASE)
        return out_matrix


if __name__ == "__main__":
    # Running this module will run tests
    # Scheme's params
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
    # Polynomial modulus
    poly_mod = np.array([1] + [0] * (n - 1) + [1])
    # Std for encryption
    std1 = 1
    # generate public and secret keys
    pk, sk = keygen(n, q, poly_mod, std1)

    params = (pk, n, q, t, poly_mod, std1)

    int_matrix = [[1, 2], [3, 4]]
    int_matrix2 = [[1, 2], [3, 4]]

    c_1 = EncryptedMatrix()
    c_2 = EncryptedMatrix()

    c_1.encrypts(int_matrix, params)
    c_2.encrypts(int_matrix2, params)
    c_3 = c_1 + c_2

    print("ENCRYPTED MATRIX IS:")
    print(c_1.poly_matrix)
    print("CHECKING ORIGINAL MATRIX MATCHES DECRYPTED MATRIX:")
    print(int_matrix)
    decrypted_matrix = c_1.decrypt(sk)
    print(decrypted_matrix)
    if np.array_equal(int_matrix, decrypted_matrix):
        print("SUCCESS!")
    else:
        print("ERROR: FAILED!")

    print("CHECK SUM FUNCTION:")
    decrypted_matrix = c_3.decrypt(sk)
    print(decrypted_matrix)
