#include <iostream>
#include <armadillo>

#include "test_arma.h"

int test_arma()
{
  // Initialize the random generator.
  arma::arma_rng::set_seed_random();

  // Create a 4x4 random matrix and print it on the screen.
  arma::mat A = arma::randu(5, 8);
  std::cout << "A:\n" << A << "\n";

  // Multiply A with his transpose:
  std::cout << "A * A.t() =\n";
  std::cout << A * A.t() << "\n";

  // Pinv of A:
  std::cout << "pinv(A):\n" << arma::pinv(A) << "\n";

  // Create a new diagonal matrix using the main diagonal of A:
  arma::mat B = arma::diagmat(A);
  std::cout << "B:\n" << B << "\n";

  return 0;
}
