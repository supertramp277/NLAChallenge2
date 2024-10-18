#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
/*****************************************************************************************************************/
// More details and explanation are at this link:
/*****************************************************************************************************************/
using namespace Eigen;
typedef Eigen::Triplet<double> T;

// Function 1: Convert a normalized matrix to Matrix<unsigned char> with range [0,255] and output as png
void outputImage(const Matrix<double, Dynamic, Dynamic, RowMajor> &output_image_matrix, int height, int width, const std::string &path)
{
    // Convert the modified image to grayscale and export it using stbi_write_png
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> new_image_output = output_image_matrix.unaryExpr(
        [](double pixel)
        {
            return static_cast<unsigned char>(std::max(0.0, std::min(255.0, pixel * 255))); // ensure range [0,255]
        });
    if (stbi_write_png(path.c_str(), width, height, 1, new_image_output.data(), width) == 0)
    {
        std::cerr << "Error: Could not save modified image" << std::endl;
    }
    std::cout << "New image saved to " << path << "\n"
              << std::endl;
}

// Function 2: Export the vector to mtx file. And the index from 1 instead of 0 for meeting the lis input file demand.
void exportVector(VectorXd data, const std::string &path)
{
    FILE *out = fopen(path.c_str(), "w");
    fprintf(out, "%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out, "%d\n", data.size());
    for (int i = 0; i < data.size(); i++)
    {
        fprintf(out, "%d %f\n", i + 1, data(i)); // Attention! here index is from 1, same as lis demand.
    }
    std::cout << "New vector file saved to " << path << std::endl;
    fclose(out);
}

// Function 3: Export a sparse matrix by saveMarket()
void exportSparsematrix(SparseMatrix<double, RowMajor> data, const std::string &path)
{
    if (saveMarket(data, path))
    {
        std::cout << "New sparse matrix saved to " << path << std::endl;
    }
    else
    {
        std::cerr << "Error: Could not save sparse matrix to " << path << std::endl;
    }
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
    Matrix<double, Dynamic, Dynamic, RowMajor> matrix_A(height, width);

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
    Matrix<double, Dynamic, Dynamic, RowMajor> matrix_AT = matrix_A.transpose();
    Matrix<double, Dynamic, Dynamic, RowMajor> matrix_AT_A = matrix_AT * matrix_A;
    std::cout << "Euclidean Norm Of ATA Is: " << matrix_AT_A.norm() << std::endl;

    /*Question2: Solve the eigenvalue of ATA by eigen and report the two largest values*/
    // ATA is absolutely symmetric by defintion, check here to ensure no round off problems in practice
    double norm_diff = (matrix_AT_A - matrix_AT_A.transpose()).norm();
    std::cout << "Check if ATA is symmetric by norm value of its difference with transpose: "
              << norm_diff << std::endl;
    // Since ATA is symmetric we can use SelfAdjointEigenSolver
    SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic, RowMajor>> eigensolver_ATA(matrix_AT_A);
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
    // std::cout << "The all eigenvalues of ATA are:\n"
    //           << eigenvalues << std::endl;

    /*Question3: Export matrix ATA in the matrix market format, move to lis and compute the largest eigenvalue*/
    if (!saveMarket(matrix_AT_A, "./ATA.mtx"))
    {
        std::cerr << "Failed to save ATA in Matrix Market format!" << std::endl;
    }
    std::cout << "Matrix ATA has been saved in Matrix Market format as ATA.mtx" << std::endl;

    /**************************************************** end ****************************************************/

    /***************************Question 5-7 : Using the SVD module of the Eigen library***************************/
    // To be done
    /**************************************************** end ****************************************************/

    /********Question 8,9 : Create a black and white checkerboard image, report norm, then introduce noise********/
    // Question8: Create a black and white checkerboard image and calculate norm
    int blockSize = 1;               // Define the block size, the professor's demand is just 1?
    int numBlocks = 200 / blockSize; // Number of blocks, in this case it's just 200 same as pixels
    Matrix<double, Dynamic, Dynamic, RowMajor> checkerboard(200, 200);

    for (int i = 0; i < numBlocks; i++)
    {
        for (int j = 0; j < numBlocks; j++)
        {
            // Determine the color of the block
            double color = ((i + j) % 2 == 0) ? 0.0 : 1.0; // Black or White

            // Fill the block with the determined color
            for (int bi = 0; bi < blockSize; bi++)
            {
                for (int bj = 0; bj < blockSize; bj++)
                {
                    checkerboard(i * blockSize + bi, j * blockSize + bj) = color;
                }
            }
        }
    }
    std::cout << "The Euclidean Norm Of Checkerboard Matrix Is: " << checkerboard.norm() << "\n"
              << std::endl;
    outputImage(checkerboard, 200, 200, "image_checkerboard.png");

    // Question9: Introduce a noise into the checkerboard image
    Matrix<double, Dynamic, Dynamic, RowMajor> noised_checkboard(200, 200);
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

    /******************************Question 10-3 : Using the SVD to solve checkboard******************************/
    // To be done
    /**************************************************** end ****************************************************/

    // Free memory
    stbi_image_free(image_data);

    return 0;
}
