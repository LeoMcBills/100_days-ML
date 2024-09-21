// Circumference and area of a circle with radius 2.5

#include <iostream>
using namespace std;

const double pi = 3.141593;

int main()
{
	double area, circumference, radius = 1.5;
	
	area = pi * radius * radius;
	circumference = 2 * pi * radius;

	cout << "\nTo Evaluate a Circle\n" << endl;
	cout << "Radius: " << radius << endl
		<< "Circumference: " << circumference  << endl
		<< "Area: " << area << endl;

	return 0;
}
