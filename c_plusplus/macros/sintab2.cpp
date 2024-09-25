// Creates a sine function table

#include <iostream>
#include <iomanip>
#include <cmath>

#define PI          3.1415926536
#define START       0.0
#define END         (2.0 * PI)
#define STEP        (PI / 8.0)
#define HEADER      (cout << "\n *************** Sine Function Table *************\n")

int main()
{
    double x;

    HEADER;

    std::cout << setw(16) << "x" << setw(20) << "sin(x)\n" << std::endl;
    std::cout << std::setw(20) << x << std::setw(16) << sin(x) << std::endl;

    return 0;           

}