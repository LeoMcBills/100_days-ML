#include <iostream>
using namespace std;

class Car {
    public:
        string brand;
        int year;

        void display() {
            cout << "Brand: " << brand << ", Year: " << year << endl;
        }
};

int main() {
    Car myCar;
    myCar.brand = "Range Rover Autobiography";
    myCar.year = 2020;
    myCar.display();

    return 0;
}