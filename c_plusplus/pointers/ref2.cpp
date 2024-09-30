// Demonstrating functions with parameters of reference type.

#include <iostream>
#include <string>
using namespace std;

bool getClient ( string& name, long& nr);
void putClient ( const string& name, const long& nr);

int main()
{
    string clientName;
    long clientNr;

    cout << "\nTo input and output client data \n" << endl;

    if ( getClient ( clientName, clientNr))
        putClient( clientName, clientNr);

    else
        cout << "Invalid input!" << endl;

    return 0;
}

bool getClient( string& name, long& nr)
{
    cout << "\nTo input client data!\n"
        << " Name:     ";
    if (!( cin >> nr)) return false;

    return true;
}

void putClient ( const string& name, const long& nr)
{
    cout << "\n---------- Client Data -----------------\n"
         << "\n Name:    "; 
    cout << name << "\n Number: ";
    cout << nr << endl;
}