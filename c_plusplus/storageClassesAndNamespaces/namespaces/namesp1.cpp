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

