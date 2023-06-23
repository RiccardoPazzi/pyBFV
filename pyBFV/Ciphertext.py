"""
This is a utility class that allows to use the standard multiplication (*) and sum (+) operations on encrypted
ciphertext
"""
from pyBFV.BFV import *
import numpy as np

BASE = 2


class EncryptedMatrix:
    def __init__(self, from_array=None, parameters=None):
        """
        There's two different ways to create an EncryptedMatrix object: without any arguments, this will create an empty
        object which can store an integer matrix using encrypts. With an array and parameters, this will simply copy the
        passed array (which must already be encrypted) and store it alongside the parameters, creating a copy of the
        passed array.
        :param from_array:
        :param parameters:
        """
        if from_array is None:
            self.initialized = False # When false the matrix can encrypt an integer vector
            return
        self.pk, self.size, self.q, self.t, self.p, self.poly_mod, self.std1 = parameters
        self.packed_parameters = parameters
        self.shape = from_array.shape
        self.poly_matrix = from_array
        self.initialized = True

    def encrypts(self, int_vector, parameters: tuple):
        """
        Takes as input a matrix of integers and converts it into the corresponding polynomial representation using
        binary base encoding, then encrypts it with the public key provided in parameters
        :param int_vector: The integer matrix to be encrypted
        :param parameters: A tuple containing all necessary parameters for the
        encryption and encoding steps. The tuple is formed as follows: (public_key, polynomial size, integer modulo
        q, plaintext modulus t, scaling factor p, cyclotomic polynomial modulus, standard deviation for the error
        polynomial generation)
        :return: The encrypted version of the integer matrix
        """
        if self.initialized:
            print("WARNING! Overriding an encrypted array which was already stored into this Object")
        self.pk, self.size, self.q, self.t, self.p, self.poly_mod, self.std1 = parameters
        self.packed_parameters = parameters
        np_input = np.array(int_vector, dtype=np.int64)
        self.shape = np_input.shape + (2, self.size)
        self.poly_matrix = np.zeros(self.shape, dtype=np.int64)
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
            self.poly_matrix[idx + (0,)] += ct0
            self.poly_matrix[idx + (1,)] += ct1


    def dot(self, other, rlk):
        """
        Dot product between two encrypted tensors, relinearization is automatically performed
        :param other:
        :param rlk: Relinearization key tuple (rlk0, rlk1)
        :return: The dot product
        """
        # We iterate through the entire matrix except the last two dimensions which store the polynomials
        # The shape will be calculated as usual for the matrix product A x B = C where A.shape = [3,4] B.shape = [4,1]
        # Therefore C.shape will be [3,1]
        if len(self.shape) != 4:
            raise NotImplementedError("dot only supports 2D matrix product, for batched product use batched_dot2D")
        # Check the dimensions are matching
        if not (self.shape[-3] == other.shape[0]):
            raise ValueError(f"Matrix shapes {self.shape} and {other.shape} are not matching!")
        # To compute output shape eliminate last three dimensions (last two are to store the pair of polynomials, the
        # other dimension is the one which will be eliminated
        poly_matrix_shape = self.shape[:-3] + other.shape[1:-2]  # Remove last three dimensions of
        output_matrix = np.zeros(poly_matrix_shape + (2, self.size))
        # Nested multiplication loop
        # For multiplication the elements in the last dimension of the first tensor will be multiplied with the elements
        # in the first dimension of the second tensor
        for idx in np.ndindex(poly_matrix_shape):
            result_poly = np.zeros((2, self.size), dtype=np.int64)
            for index in range(self.shape[-3]):
                # Compute the indexes which will range to compute dot product between 1D slices of the tensor
                first_idx = idx[:-1] + (index,)
                second_idx = (index,) + idx[1:]
                # Extract first element which will be multiplied
                ct0_0 = self.poly_matrix[first_idx + (0,)]
                ct1_0 = self.poly_matrix[first_idx + (1,)]

                # Extract second polynomial of the multiplication
                ct0_1 = other.poly_matrix[second_idx + (0,)]
                ct1_1 = other.poly_matrix[second_idx + (1,)]

                # Multiply encrypted polynomials
                mul_poly = mul_cipher_v2((ct0_0, ct1_0), (ct0_1, ct1_1), self.q, self.t, self.p, self.poly_mod, rlk[0], rlk[1])

                # Sum all the mul poly into one polynomial
                result_poly = add_cipher(result_poly, mul_poly, self.q, self.poly_mod)

            # Extract the two polynomials in the resulting ciphertext
            ct0 = result_poly[0]
            ct1 = result_poly[1]

            # Pad with 0 results to always match the complete size of the polynomials
            ct0 = np.pad(ct0, (0, self.size - len(ct0)), constant_values=0)
            ct1 = np.pad(ct1, (0, self.size - len(ct1)), constant_values=0)
            # Update output_matrix accordingly
            output_matrix[idx + (0,)] += ct0
            output_matrix[idx + (1,)] += ct1

        # Return EncryptedMatrix object containing the output matrix
        e_result = EncryptedMatrix(output_matrix, self.packed_parameters)
        return e_result

    def scale(self, encrypted_scale_factor: np.ndarray, rlk):
        """
        Scales the input matrix using a scale factor encrypted under the same key pk
        :param encrypted_scale_factor: encrypted integer representing the scale factor
        :param rlk: Relinearization key
        :return: encrypted scaled matrix
        """
        poly_shape = self.shape[:-2]
        result = np.zeros(self.shape)
        for idx in np.ndindex(poly_shape):
            # Extract ciphertexts from matrix
            ct0 = self.poly_matrix[idx + (0,)]
            ct1 = self.poly_matrix[idx + (1,)]
            scale0 = encrypted_scale_factor[0]
            scale1 = encrypted_scale_factor[1]
            # Multiply with scale factor
            scaled_ct0, scaled_ct1 = mul_cipher_v2((ct0, ct1), (scale0, scale1), self.q, self.t, self.p, self.poly_mod,
                                                   rlk[0], rlk[1])

            # Add ciphertexts to result, once more we need to pad with zeroes
            scaled_ct0 = np.pad(scaled_ct0, (0, self.size - len(scaled_ct0)), constant_values=0)
            scaled_ct1 = np.pad(scaled_ct1, (0, self.size - len(scaled_ct1)), constant_values=0)
            result[idx + (0,)] += scaled_ct0
            result[idx + (1,)] += scaled_ct1
        # Convert result into EncryptedMatrix
        e_result = EncryptedMatrix(result, self.packed_parameters)
        return e_result

    def __rmul__(self, other):
        if not (isinstance(other, int) or isinstance(other, np.int64)):
            print("ERROR! Cannot use * operator with elements other than integers")
            return self
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
    n = 2 ** 4
    # ciphertext modulus, MODULUS parameter
    q = 2 ** 25
    # plaintext modulus, maximum value for plaintext in the worst case
    # Some operations might still produce the expected results when the number exceeds t due to base decomposition
    # However t for sure limits the number of sums to t, because even with binary decomposition the 1's added will
    # eventually surpass e.g. 128
    # Limiting the number of sums to t will in turn limit the dot product which can only operate on tensors of the form
    # (.. x t) . (t x ..) where t is the dimension which will be eliminated from both tensors
    t = 128
    # base for relin_v1
    T = int(np.sqrt(q))
    # modulus switching modulus
    p = q
    # Polynomial modulus
    poly_mod = np.array([1] + [0] * (n - 1) + [1])
    # Std for encryption
    std1 = 1
    std2 = 1
    # generate public and secret keys
    pk, sk = keygen(n, q, poly_mod, std1)
    rlk = evaluate_keygen_v2(sk, n, q, poly_mod, p, std2)

    params = (pk, n, q, t, p, poly_mod, std1)

    int_matrix = [[1, 2], [3, 4]]
    int_matrix2 = [[1, 2], [3, 4]]
    int_vector = [[1], [5]]  # Column vector

    c_1 = EncryptedMatrix()
    c_2 = EncryptedMatrix()
    cv_1 = EncryptedMatrix()
    scale = 5
    scale_poly = int2base(scale, BASE)
    e_scale = encrypt(pk, n, q, t, poly_mod, scale_poly, std1)

    c_1.encrypts(int_matrix, params)
    c_2.encrypts(int_matrix2, params)
    cv_1.encrypts(int_vector, params)
    c_3 = c_1 + c_2

    print("CHECKING ORIGINAL MATRIX MATCHES DECRYPTED MATRIX:")
    decrypted_matrix = c_1.decrypt(sk)
    print(decrypted_matrix)
    if np.array_equal(int_matrix, decrypted_matrix):
        print("SUCCESS!")
    else:
        print("ERROR: FAILED!")

    print("\n---------------------------------\n")  # Visual separator

    print("CHECK SUM FUNCTION:")
    decrypted_matrix = c_3.decrypt(sk)
    print(decrypted_matrix)
    if np.array_equal(np.array(int_matrix)+np.array(int_matrix2), decrypted_matrix):
        print("SUCCESS!")
    else:
        print("ERROR: FAILED!")

    print("\n---------------------------------\n")  # Visual separator

    print("DOT PRODUCT CHECK:")
    c_4 = c_1.dot(c_2, rlk)
    decrypted_matrix = c_4.decrypt(sk)
    print(decrypted_matrix)
    if np.array_equal(np.array(int_matrix).dot(np.array(int_matrix2)), decrypted_matrix):
        print("SUCCESS!")
    else:
        print("ERROR: FAILED!")

    print("\n---------------------------------\n")  # Visual separator

    print("SCALING CHECK:")
    c_6 = c_1.scale(e_scale, rlk)
    decrypted_matrix = c_6.decrypt(sk)
    print(decrypted_matrix)
    if np.array_equal(scale * np.array(int_matrix), decrypted_matrix):
        print("SUCCESS!")
    else:
        print("ERROR: FAILED!")

    print("\n---------------------------------\n")  # Visual separator

    print("MATRIX-VECTOR DOT PRODUCT CHECK:")
    c_5 = c_1.dot(cv_1, rlk)
    decrypted_matrix = c_5.decrypt(sk)
    print(decrypted_matrix)
    if np.array_equal(np.array(int_matrix).dot(np.array(int_vector)), decrypted_matrix):
        print("SUCCESS!")
    else:
        print("ERROR: FAILED!")
