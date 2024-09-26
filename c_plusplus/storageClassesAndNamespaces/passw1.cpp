// The functions getPassword() and timediff() to read and examine a password

#include <iostream>
#include <iomanip>
#include <ctime>
#include <string>
using namespace std;

long timediff(void);
bool getPassword();
static string secret = "ISUS";
static long maxcount = 3, maxtime = 60;

int main()
{
    if ( getPassword() )
        cout << "Access granted\n";
    else
        cout << "Access denied\n";

    return 0;
}

bool getPassword()
{
    bool ok_flag = false;
    string word;
    int count = 0, time = 0;
    timediff();

    while ( ok_flag != true && ++count <= maxcount)
    {
        cout << "\n\nInput the password: ";
        cin.sync();
        cin >> setw(20) >> word;
        time += timediff();

        if ( time >= maxtime )
        {
            cout << "Time expired\n";
            break;
        }

        if ( word != secret )
        {
            cout << "Wrong password\n";
            cout << "You have " << maxcount - count << " more tries\n";
        }
        else
        {
            cout << "Correct password\n";
            ok_flag = true;
        }
    }

    return ok_flag;
}

long timediff()
{
    static long sec = 0;
    long oldsec = sec;
    time( &sec);
    return (sec - oldsec);
}