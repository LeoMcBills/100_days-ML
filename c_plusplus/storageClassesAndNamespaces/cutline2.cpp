// Containing the function cutline(), which removes tabulator characters at the end of the string line.
// The string line has to be globally defined in another source file.

#include <string>
using namespace std;

extern string line;

void cutline()
{
    int i = line.size();

    while ( i-- >= 0 )
        if ( line[i] == ' ' && line[i] != '\t')
            break;

    line.resize(++i);
}