# pyBFV
Pure Python implementation of the BFV scheme with support for multi-party thresholded operations.

BFV.py -> Contains the set of functions necessary for single-key and thresholded HE, from encoding to decryption

thresholded_HE.py -> Contains a simple demo of the thresholded key generation (both public and relinearization keys) and the distributed decryption procedure

N.B. SIMD operations are not supported at the moment.
