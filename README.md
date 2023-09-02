# pyBFV
Pure Python implementation of the homomorphic encryption BFV scheme with support for multi-party thresholded operations and matrix operations using the EncryptedMatrix class.

#### BFV.py
Contains the set of functions necessary for single-key and thresholded HE, from encoding to decryption

#### BFV_bigint.py
Contains the same methods as BFV but adapted to work with python big integers, bypassing the 64 bit limit

#### thresholded_HE.py
Contains a simple demo of the thresholded key generation (both public and relinearization keys) and the distributed decryption procedure

#### thresholded_Matrix.py
Contains a simple demo of encrypted multi-party matrix multiplication, no SIMD capability unfortunately

#### bigint_experiment.py
Simple operations using the BFV_bigint module


### WARNING! 
#### This library is a demo, do not use it in applications where security is necessary.
Even though it's a faithful implementation of BFV some of the methods used for performance and practicality
reasons are not designed to be secure (e.g. the random library).

### CREDITS
The implementation of thresholded functionalities, matrix operations and the translation to Python big integers is my own.
However for the basic BFV functionalities and relinearization I used the work from:
https://gist.github.com/youben11/f00bc95c5dde5e11218f14f7110ad289
and
https://bit-ml.github.io/blog/post/homomorphic-encryption-toy-implementation-in-python/

Which saved me a lot of time since everything didn't have to be implemented from scratch :)
