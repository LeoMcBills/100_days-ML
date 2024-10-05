#include <iostream>

int main(int argc, char **argv) {
    // Flush after every std::cout / std::cerr
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    std::cout << "This is standard output (cout)." << std::endl;
    std::cerr << "This is standard error (cerr)." << std::endl;

    std::cout << "Performing some calculations..." << std::endl;
    int a = 5, b = 10;
    int result = a + b;
    std::cout << "Result of 5 + 10: " << result << std::endl;

    return 0;
}
