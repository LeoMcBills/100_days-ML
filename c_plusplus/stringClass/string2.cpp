// Reads several lines of text and outputs in reverse order.

#include <iostream>
#include <string>
using namespace std;

string prompt("Please enter some text!\n"),
        line( 50, '-' );

int main() {
    prompt += "Terminate the input with an empty line.\n";
    cout << line << '\n' << prompt << line << endl;
    string text, line;

    while (true) {
        getline( cin, line);
        if ( line.length() == 0)
            break;

        text = line + '\n' + text;
    }

    cout << line << '\n'
        << "Your lines of text in reverse order:"
        << '\n' << line << endl;
    cout << text << endl;

    return 0;
}