>**Storage Class Specifiers**
The storage class of an object is determined by
- the position of its declaration in the source file
- the storage class specifier, which can be supplied optionally.

The following storage class specifiers can be used  
<pre>
extern          static          auto            register
</pre>

Essentially, an object is only available after you have declared it within a *translation unit*. A translation unit, also referred to as module, comprises the source file you are compiling and any header files you have included.

