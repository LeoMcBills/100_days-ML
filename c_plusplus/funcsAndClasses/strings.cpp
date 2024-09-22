// To use strings

#include <iostream>
#include <string>

using namespace std;

int main() 
{
    // Defines four strings:
    string promt("What is your name: "),
    name,
    line(40, '-'),
    total = "Hello ";

    cout << promt,  // Request for input
    getline(cin, name);
    total = total + name;

    cout << line << endl
         << total << endl;

    cout << "Your name is " << name.length() << " characters long!" << endl;

    cout << line << endl;

    return 0;
}