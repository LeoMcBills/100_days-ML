// Definition and use of variables
//
#include <iostream>
using namespace std;

int gVar1;		// Globak variablees
int gVar2 = 2;		// Explicit initialization

int main()
{
	char ch('A');	// Local variable being initialized or one could just explicitly initialize it as char ch = 'A';
	
	cout << "Value of gVar1: " << gVar1 << endl;
	cout << "Value of gVar2: " << gVar2 << endl;
	cout << "Character in ch: " << ch << endl;

	int sum, number = 3; // local variables with and without explicit initialization

	sum = number + 5;
	cout << "Value of sum: " << sum << endl;

	return 0;
}
