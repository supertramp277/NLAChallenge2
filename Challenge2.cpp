#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
/*****************************************************************************************************************/
// More details and explanation are at this link: https://github.com/supertramp277/NLAChallenge2
/*****************************************************************************************************************/
using namespace Eigen;
typedef Eigen::Triplet<double> T;

// Function 1: Convert a normalized matrix to Matrix<unsigned char> with range [0,255] and output it as png
void outputImage(const MatrixXd &output_image_matrix, int height, int width, const std::string &path)
{
    // Convert the modified image to grayscale and export it using stbi_write_png, just need this stb related to be rowmajor
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> new_image_output = output_image_matrix.unaryExpr(
        [](double pixel)
        {
            return static_cast<unsigned char>(std::max(0.0, std::min(255.0, pixel * 255))); // ensure range [0,255]
        });
    if (stbi_write_png(path.c_str(), width, height, 1, new_image_output.data(), width) == 0)
    {
        std::cerr << "Error: Could not save modified image" << std::endl;
    }
    std::cout << "New image saved to " << path << std::endl;
}

/*--------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------Main()-------------------------------------------------------*/
/*--------------------------------------------------------------------------------------------------------------*/

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char *input_image_path = argv[1];

    /*************************************Load the image by using stb_image*************************************/
    int width, height, channels;
    // for greyscale images force to load only one channel
    unsigned char *image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data)
    {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }

    std::cout << "Image " << argv[1] << " loaded: " << height << "x" << width << " pixels with " << channels << " channels" << std::endl;

    /*************Convert the image_data to MatrixXd form, each element value is normalized to [0,1]*************/
    // Attention! We use RowMajor notation!
    MatrixXd matrix_A(height, width);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = (i * width + j) * channels;
            matrix_A(i, j) = static_cast<double>(image_data[index]) / 255; // normalized value range from 0 to 1
        }
    }

    // Report the size of the matrix
    std::cout << "The original image " << argv[1] << " in matrix form has dimension: " << matrix_A.rows() << " rows x " << matrix_A.cols()
              << " cols = " << matrix_A.size() << "\n"
              << std::endl;
    /**************************************************** end ****************************************************/

    /**********************************Question 1-4 : Solve  Eigenvalue Problems**********************************/
    /*Question1: Compute the Euclidean norm (Frobenius norm) of ATA*/
    std::cout << "---------------Part1: Solve Eigenvalue Problems Of ATA---------------" << std::endl;
    MatrixXd matrix_AT = matrix_A.transpose();
    MatrixXd matrix_AT_A = matrix_AT * matrix_A;
    std::cout << "Euclidean Norm Of ATA Is: " << matrix_AT_A.norm() << std::endl;
    /*Question2: Solve the eigenvalue of ATA by eigen and report the two largest values*/
    // ATA is absolutely symmetric by defintion, check here to ensure no round off problems in practice
    double norm_diff = (matrix_AT_A - matrix_AT_A.transpose()).norm();
    std::cout << "Check if ATA is symmetric by norm value of its difference with transpose: "
              << norm_diff << std::endl;
    // Since ATA is symmetric we can use SelfAdjointEigenSolver
    SelfAdjointEigenSolver<MatrixXd> eigensolver_ATA(matrix_AT_A);
    if (eigensolver_ATA.info() != Success)
    {
        std::cerr << "Eigenvalue computation failed!" << std::endl;
        return -1;
    }
    // Get the vector and sort it by descent to find largest 2 eigenvalues
    VectorXd eigenvalues = eigensolver_ATA.eigenvalues();
    // std::sort(eigenvalues.data(), eigenvalues.data() + eigenvalues.size(), std::greater<double>());
    std::cout << "The two largest eigenvalues of ATA are:\n"
              << eigenvalues[eigenvalues.size() - 1] << "\n"
              << eigenvalues[eigenvalues.size() - 2] << std::endl;
    saveMarketVector(eigenvalues, "./AllEigenvaluesOfATA.txt");

    /*Question3: Export matrix ATA in the matrix market format, move to lis and compute the largest eigenvalue*/
    if (!saveMarket(matrix_AT_A, "./ATA.mtx"))
    {
        std::cerr << "Failed to save ATA in Matrix Market format!" << std::endl;
    }
    std::cout << "Matrix ATA has been saved in Matrix Market format as ATA.mtx\n"
              << std::endl;

    /**************************************************** end ****************************************************/

    /***************************Question 5-7 : Using the SVD module of the Eigen library***************************/
    // Question5: Compute the thin SVD of A, diagonal matrix sigma's norm and check
    std::cout << "------------------Part2: Perform SVD Of A------------------" << std::endl;
    BDCSVD<MatrixXd> svd(matrix_A, ComputeThinU | ComputeThinV); // thin-solver
    MatrixXd U = svd.matrixU();
    MatrixXd Sigma = svd.singularValues().asDiagonal();
    MatrixXd V = svd.matrixV();
    MatrixXd matrix_SVD_A = U * Sigma * V.transpose();
    std::cout << "Euclidean norm of the diagonal matrix sigma:" << Sigma.norm() << std::endl;
    std::cout << "Check If SVD Is Right Done By Diff Norm:" << (matrix_A - matrix_SVD_A).norm() << std::endl;
    saveMarket(U, "./MatrixU.mtx");
    saveMarket(Sigma, "./MatrixSigma.mtx");
    saveMarket(V, "MatrixV.mtx");
    saveMarketVector(svd.singularValues(), "./SingularValuesOfA.txt");
    /**
     * Question6: Compute the matrices C and D, assuming k = 40 and k = 80.
     * And Report the number of nonzero entries in the matrices C and D.
     * Reason just k=40 will work very well? Because the eigenvalues are the largest 40
     * So most contributions are from these eigenpairs, others contribute very few */
    MatrixXd C40 = U.leftCols(40);
    MatrixXd D40 = V.leftCols(40) * svd.singularValues().head(40).asDiagonal();
    std::cout << "Nonzero Entries For Matrix C40: " << C40.nonZeros() << std::endl;
    std::cout << "Nonzero Entries For Matrix D40: " << D40.nonZeros() << std::endl;
    // Compute the matrices C and D For k=80 again
    MatrixXd C80 = U.leftCols(80);
    MatrixXd D80 = V.leftCols(80) * svd.singularValues().head(80).asDiagonal();
    std::cout << "Nonzero Entries For Matrix C80: " << C80.nonZeros() << std::endl;
    std::cout << "Nonzero Entries For Matrix D80: " << D80.nonZeros() << std::endl;
    // Question7: Compute the compressed images as the matrix product CD^T (again for k = 40 and k = 80)
    MatrixXd CompressedA40 = C40 * D40.transpose();
    std::cout << "A40's Rows:" << CompressedA40.rows() << "\t" << "Cols:" << CompressedA40.cols() << std::endl;
    outputImage(CompressedA40, CompressedA40.rows(), CompressedA40.cols(), "./image_compressed_k40.png");
    // again for k=80
    MatrixXd CompressedA80 = C80 * D80.transpose();
    std::cout << "A80's Rows:" << CompressedA80.rows() << "\t" << "Cols:" << CompressedA80.cols() << std::endl;
    outputImage(CompressedA80, CompressedA80.rows(), CompressedA80.cols(), "./image_compressed_k80.png");
    /**************************************************** end ****************************************************/

    /********Question 8,9 : Create a black and white checkerboard image, report norm, then introduce noise********/
    // Question8: Create a black and white checkerboard image and calculate norm
    std::cout << "\n------------------Part3: Checkboard Related Problems------------------" << std::endl;
    int blockSize = 25;              // Define the block size, the professor's demand is just 1?
    int numBlocks = 200 / blockSize; // Number of blocks, in this case it's just 200 same as pixels
    MatrixXd checkerboard(200, 200);
    for (int i = 0; i < numBlocks; i++)
    {
        for (int j = 0; j < numBlocks; j++)
        {
            // Determine the color of the block
            double color = ((i + j) % 2 == 0) ? 0.0 : 1.0; // Black or White

            // Fill the block with the determined color,
            // bi and bj is in block index and need to also consider outter position
            for (int bi = 0; bi < blockSize; bi++)
            {
                for (int bj = 0; bj < blockSize; bj++)
                {
                    checkerboard(i * blockSize + bi, j * blockSize + bj) = color;
                }
            }
        }
    }
    std::cout << "The Euclidean Norm Of Checkerboard Matrix Is: " << checkerboard.norm()
              << std::endl;
    outputImage(checkerboard, 200, 200, "image_checkerboard.png");
    // Question9: Introduce a noise into the checkerboard image
    MatrixXd noised_checkboard(200, 200);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(-50, 50);
    for (int i = 0; i < 200; i++)
    {
        for (int j = 0; j < 200; j++)
        {
            int noise = distribution(generator);
            double noisedata = checkerboard(i, j) + static_cast<double>(noise) / 255; // Normalized
            noised_checkboard(i, j) = std::max(0.0, std::min(1.0, noisedata));
        }
    }
    outputImage(noised_checkboard, 200, 200, "image_noised_checkerboard.png");
    /**************************************************** end ****************************************************/

    /******************************Question 10-12 : Using the SVD to solve checkboard******************************/
    // Question10: Perform svd of the matrix of noisy image. Report the two largest computed singular values.
    BDCSVD<MatrixXd> checkboard_svd(noised_checkboard, ComputeThinU | ComputeThinV);
    VectorXd singularValuesCheckboard = checkboard_svd.singularValues();
    std::cout << "The largest two singular values of noised checkboard are:\n"
              << singularValuesCheckboard[0] << "\n"
              << singularValuesCheckboard[1] << std::endl;
    saveMarketVector(singularValuesCheckboard, "./SingularValuesOfCB.txt");
    // Question11: Create the matrices C and D, assuming k = 5 and k = 10. Report the size of the matrices C and D.
    MatrixXd C5 = checkboard_svd.matrixU().leftCols(5);
    MatrixXd D5 = checkboard_svd.matrixV().leftCols(5) * singularValuesCheckboard.head(5).asDiagonal();
    std::cout << "Size Of Matrix C When K=5:\n"
              << "Rows: " << C5.rows() << "\n"
              << "Cols: " << C5.cols() << std::endl;
    std::cout << "Size Of Matrix D When K=5:\n"
              << "Rows: " << D5.rows() << "\n"
              << "Cols: " << D5.cols() << std::endl;
    // Also cumpute result when k=10
    MatrixXd C10 = checkboard_svd.matrixU().leftCols(10);
    MatrixXd D10 = checkboard_svd.matrixV().leftCols(10) * singularValuesCheckboard.head(10).asDiagonal();
    std::cout << "Size Of Matrix C When K=10:\n"
              << "Rows: " << C10.rows() << "\n"
              << "Cols: " << C10.cols() << std::endl;
    std::cout << "Size Of Matrix D When K=10:\n"
              << "Rows: " << D10.rows() << "\n"
              << "Cols: " << D10.cols() << std::endl;
    // Question12: Compute the compressed images as the matrix product CD^T (again for k = 5 and k = 10)
    MatrixXd CompressedCB5 = C5 * D5.transpose();
    outputImage(CompressedCB5, CompressedCB5.rows(), CompressedCB5.cols(), "./image_compressed_CB_k5.png");
    // again for k=10
    MatrixXd CompressedCB10 = C10 * D10.transpose();
    outputImage(CompressedCB10, CompressedCB10.rows(), CompressedCB10.cols(), "./image_compressed_CB_k10.png");
    /**************************************************** end ****************************************************/

    // Free memory
    stbi_image_free(image_data);

    return 0;
}
