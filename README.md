# NLA Challenge2 Results Group 6

## Output Files

- [`output.txt`](output.txt): Contains the output resultls of the main cpp program.
- [`output_lis.txt`](output_lis.txt) : Contains the output results from lis for computing the largest eigenvalue of $A^TA$.

- [`output_lis_shift.txt`](output_lis_shift.txt) : Contains the output results from lis about solving the smallest eigenvalue of $(A-\mu I)^T(A-\mu I)$ which also means solving largest eigenvalue of $A^TA$ by shifting for acceleration
  - $\mu = 1.608300e+04$
  - $\text{Inverse: number of iterations} = 3$
  - $\text{Inverse: eigenvalue} = 1.608332e+04$
- Some other files like  `AllEigenvaluesOfATA.txt`, `MatrixSigma.mtx`, `SingularValuesOfA.txt` ... are used for own checking.

## Some Comments

About the last question, we can see that by this compressed way through svd, we have decreased the noise difference from 16 to about 12 successfully.

- however, it seems when $k=10$ the compressed image compared to the original one is less denoised that the case for $k=5$
![alt text](screenshot.png)
