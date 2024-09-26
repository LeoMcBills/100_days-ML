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

