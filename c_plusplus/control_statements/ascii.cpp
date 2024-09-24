// To output an ASCII code table

#include <iostream>
#include <iomanip>
using namespace std;

int main()
{
    int ac = 32, upper;
    char answer;

    while (true)
    {
        cout << "\nCharacter    Decimal     Hexadecimal\n\n" << endl;

        for (upper = ac + 20; ac < upper && ac < 256; ++ac){
            cout << "   " << (char)ac << setw(10) << "dec" << setw(10) << "hex" << ac << endl;
        }

        if (upper >= 256)   break;

        cout << "\nGo on -> <return>, stop -> <q> + <return>" << endl;

        cin.get(answer);
        
        if (answer == 'q' || answer == 'Q') break;
        
        cin.sync();
    }
    
    return 0;
}