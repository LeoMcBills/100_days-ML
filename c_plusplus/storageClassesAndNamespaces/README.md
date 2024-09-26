>**Storage Class Specifiers**
The storage class of an object is determined by
- the position of its declaration in the source file
- the storage class specifier, which can be supplied optionally.

The following storage class specifiers can be used  
<pre>
extern          static          auto            register
</pre>

Essentially, an object is only available after you have declared it within a *translation unit*. A translation unit, also referred to as module, comprises the source file you are compiling and any header files you have included.

As a programmer, you can define an object with:  
* block scope  
* file scope  
* program scope  

The object is only available in the code block in which it was
defined. The object is no longer visible once you have left
the code block.
The object can be used within a single module. Only the
functions within this module can reference the object. Other
modules cannot access the object directly.
The object is available throughout the program, providing a
common space in memory that can be referenced by any pro-
gram function. For this reason, these objects are often
referred to as global.
Access to an object as defined by the object’s storage class is independent of anyaccess controls for the elements of a class. Namespaces that subdivide program scope and
classes will be introduced at a later stage.  

## Lifetime  
Objects with block scope are normally created automatically within the code block that defines them. Such objects can only be accessed by statements within that block and are called **local** to that block. The memory used for these objects is freed after leaving the code block. In this case, the lifetime of the objects is said to be **automatic**.

However, it is possible to define objects with block scope that are available through-out the runtime of a program. The lifetime of these objects is said to be **static**. When the program flow re-enters a code block, any pre-existing conditions will apply.

Objects with program and file scope are always **static**. These objects are created when a program is launched and are available until the program is terminated.

Four storage classes are available for creating objects with the scope and lifetime

