// Defines and tests namespaces.
// ---------------------------------------------------
#include <string>

namespace MySpace
{
    std::string mess = "Within namespace MySpace";
    int count = 0;          // Definition: MySpace::count
    double f( double);      // Prototype:  MySpace::f()
}

namespace YourSpace
{
    std::string mess = "Within namespace YourSpace";
    void f()
    {
        mess += '!';        // Definition: YourSpace::mess
    }
}

namespace MySpace           // Back in MySpace
{
    int g(void);           // Prototype:  MySpace::g()
    double f( double y)    // Definition: MySpace::f()
    {
        return y / 10.0;
    }
}

int MySpace::g()
{
    return ++count;        // Separate definition of MySpace::g()
}

#include <iostream>        // cout, ... within namespace std
int main()
{

    
}