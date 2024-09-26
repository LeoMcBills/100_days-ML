// A filter to remove white space characters at the ends of lines.
#include <iostream>
#include <string>
using namespace std;

void cutline( void );
string line;

int main()
{
    while (getline(cin, line))
    {
        cutline();
        cout << line << endl;
    }
    return 0;
}