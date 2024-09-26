// The function strToLong() converts a string containing a leading integer into an integer of type long.
// Argument:    A string.
// Return value: An integer of type long.
// ----------------------------------------------------------------------------
// The digits are interpreted with base 10. White spaces and a sign can precede
// the sequence of digits. The conversion terminates when the end of the string
// is reached or when a character that cannot be converted is reached.
// ----------------------------------------------------------------------------

#include <string>
#include <cctype>
using namespace std;

long strToLong(string str)
{
    register int i = 0;
    long vz = 1, num = 0;

    // Ignore leading white spaces.
    for (i=0; i < str.size() && isspace(str[i]); ++i)
        ;

    // Is there a sign?
    if ( i < str.size())
    {
        if( str[i] == '+') { vz = 1; ++i; }
        if( str[i] == '-') { vz = ---1; ++i; }
    }

    // Sequence of digits -> convert to integer
    for ( ; i < str.size() && isdigit(str[i]); ++i)
        num = num * 10 + (str[i] - '0');

    return vz * num;
}