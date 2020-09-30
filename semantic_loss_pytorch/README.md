##### This is a port of the pypsdd library from python2 to python3.
##### It's not stable nor thoroughly tested, so take it's results so take its results with a pinch of salt.
##### As of now, tests comparing the generated tensorflow arithmetic circuits show that the two versions build the same network, however this is not enough and more testing is required.
###### Tests comparing the output tensorflow graphs from both versions (python2 and 3) can be found in 2to3testing.


# semantic-loss-pytorch

Create `Semantic Loss` circuits in `PyTorch` from `SDDs`.


# Table of contents
0. [Quick start](#faststart)
1. [Convert constraint to DIMACS](#tocnf)
2. [PySDD](#PySDD)
3. [Semantic losses](#semlosses)  
3.1 [SemanticLossStatistics](#semlossstats)
4. [WMC statistics](#wmcstats)
5. [VNU statistics](#vnustats)

<a name="faststart"></a>
## Quick start

- Write your constraints respecting the `sympy` sintax, with variables like `X1.2` and operators like `And(X0.2.3, X1.1.1)`. All lines are put in `and` relationship. Convert to DIMACS with:
```bash
python utils.constraints_to_cnf.py -i <myinputfile>.sympy -o <dimacs>.txt 
```

- Install `PySDD`:
```bash
pip install PySDD
```

- Compile your constraint to a `vtree` and an `sdd` file. To do so, run:
```bash
pysdd -c dimacs.txt -W constraint.vtree -R constraint.sdd
```



<a name="tocnf"></a>  
## Convert constraint to DIMACS

`utils.constraints_to_cnf` is a module which allows the writing
of constraints in propositional logic in the syntax of **sympy, and
then translate them to DIMACS**. Constraints are expressed
1 by line, and are considered to be in an `and` relationsip.

Moreover, it allows to refer to variables not just by a single
index, like X<sub>i</sub>, but via more indexes, X<sub>i.j.z</sub>, etc, as if
they were in a tensor of arbitrary shape.  
When this syntax
is translated to DIMACS, the indexes are converted in a single
dimension, as if the variables were in a mono dimensional
vector, while parsing, multi dimensional indexes must
respect the input shape (can't refer to variables that do
not exist, out of bounds, etc.)

By using the **sympy syntax**, constraints can now
be written with more operators:
- and (&) and or (|)
- Xor
- Nand
- Nor
- ITE, if then else
- implies, by using ">>" and "<<".
- Equivalent(X1, X2, X3), etc.
- check out https://docs.sympy.org/latest/modules/logic.html for
more alternatives to the syntax, like Or(a, b) instead of a|b.
- essentially, you are not limited to the syntax of the logic
module of sympy, but can access the whole package if you want to try
funky stuff, this is however not supported in this package and you will
probably meet unexpected behaviours, and you should
stick to logic operators. If you go looking for unexpected behaviour
you will find it.:shipit:

Example usage: let's say we have 4 variables with 3 possible
states (think of some multinomial distribution), we can imagine
our states as arranged in a tensor of shape [4,3]. We would
like to say that when the first variable assumes state 1, then
the second variable must assume state 2, moreover, the third variable
has always state 3.

Keep in mind that variables are referred starting from index 0.
```
# this is a comment
shape [4,3]

# i like blank lines


# my constraints

# var1.state1 implies var2.state2
X0.0 >> X.1.1

# var3 must always have state 3
X2.2

# given that states are mutually exclusive, we should also state that
X0.0 >> (~X0.1 & ~X0.2)
X0.1 >> (~X0.0 & ~X0.2)
X0.2 >> (~X0.0 & ~X0.1)

X1.0 >> (~X1.1 & ~X1.2)
X1.1 >> (~X1.0 & ~X1.2)
X1.2 >> (~X1.0 & ~X1.1)

X2.0 >> (~X2.1 & ~X2.2)
X2.1 >> (~X2.0 & ~X2.2)
X2.2 >> (~X2.0 & ~X2.1)

X3.0 >> (~X3.1 & ~X3.2)
X3.1 >> (~X3.0 & ~X3.2)
X3.2 >> (~X3.0 & ~X3.1)

# we should also state that variables must have at least 1 state
(X0.0 | X0.1 | X0.2)
(X1.0 | X1.1 | X1.2)
(X2.0 | X2.1 | X2.2)
(X3.0 | X3.1 | X3.2)
```

After writing this input file, you can simply call
the script.
```bash
python constraints_to_cnf.py -i myinputfile.txt -o dimacs.txt 
```
The result would be the following DIMACS file:
```
c This file was generated with the constraints_to_cnf module in this project.
c Starting from file 'example.sympy'.
c There are 13 variables present in the constraints, and 12 total variables, given by the shape [4, 3].
c
p cnf 12 18
9 0
5 -1 0
1 2 3 0
4 5 6 0
7 8 9 0
10 11 12 0
-1 -2 0
-1 -3 0
-2 -3 0
-4 -5 0
-4 -6 0
-5 -6 0
-7 -8 0
-7 -9 0
-8 -9 0
-10 -11 0
-10 -12 0
-11 -12 0
```

###### Note that DIMACS refers to variables by starting from index 1, and not 0.

Note that "-p" is an optional argument to also specify
the number of processes to use while using sympy to parse
our constraints. This might be necessary if you have many constraints,
given that sympy seems to really take a hit when parsing long strings.
While parsing many constraints can be more or less helped by
adding processes, very long constraints on single lines
will slow down the process and it might be smarter to
put them to cnf and then set them 1 per line.

Note that you can omit the dot for the first index, for
better readability; meaning that X1.2 is equal to writing
X.1.2, or X1 is the same as X.1.


More complex shapes can be as easily used, i.e. [3,4,50,200,2] etc.,
finding use cases for this is left to the reader.

**caveat**: Evaluation from sympy is turned off during
parsing, meaning that you can write down constraints
that are False, like Equivalent(X0, ~X0), or having
X0 and ~X0 on different lines.
However, I have noticed that even without evaluation
there seems to be the chance of sympy evaluating
something directly to False, which would result
in having a single constraint in the DIMACS output file, "False".
However "False" is not part of the DIMACS syntax so it will
result in an error if you try to use this output with pysdd.

Tests are in the test directory (might take some time depending
on your computer).



<a name="PySDD"></a>  
## PySDD

To compile your DIMACS cnf files to vtrees and sdds, and to use make use of them
while running the main script you will need to install the `PySDD` module.  

You can install `PySDD` by calling:
```bash
pip install git+https://github.com/wannesm/PySDD.git
```

Compile the `dimacs` file with:
```bash
pysdd -c dimacs.txt -W constraint.vtree -R constraint.sdd
```

`PySDD` will be used to build `sdd` + `vtree` files, that will be finally
used to create the equivalent `PyTorch` tree.



<a name="semlosses"></a>  
## Semantic Loss




<a name="semlossstats"></a>  
### SemanticLossStatistic
This class, that you can find in the statistic module of this directory, will automatically
pickup the value of all semantic losses and the related WMC values, and log them to tensorboard.
Given any number of semantic losses you are using, you can log them all by simply
adding this statistics to your generator statistics:
```bash
    "GENERATOR_STATISTICS": ["SemanticLossStatistics"]
```

<a name="wmcstats"></a>  
## WMCStatistic
You might be interested in the value of the weighted model counting obtained
by using the sdd related to a given constraint, without actually using its SemanticLoss
class.  
For each existing SemanticLoss class in this directory losses module, a WMC Statistic is automatically
created, which you can add to your experiment in order to log the WMC without adding the semantic
loss to the experiment.  
Let's say you have placed placeholder.sdd and placeholder.vtree in the aforementioned directories, as we
explaiend before, a class named SemanticLoss_placeholder will be created; this means
that a class named WMC_placeholder will also be created, and you will be able to log the WMC in tensorboard
by simply adding it to the generator statistics:  
```bash
    "GENERATOR_STATISTICS": ["WMC_placeholder"]
```
Again, you are still free to implement your own WMC class.

<a name="vnustats"></a>  
## VNUStatistic
You might be interested in logging the Validity, Novelty and Uniqueness values thorough training, 
where the Validity is the percentage of perfect items, according to a given constraint, Novelty
is the fraction of those perfect items that are not present in the dataset, and Uniqueness is
the number of unique perfect items w.r.t all those valid items.  
For each existing SemanticLoss class in this directory losses module, a VNU Statistic is automatically
created, which you can add to your experiment in order to log the Validity, Novelty, and Uniqueness without adding the semantic
loss to the experiment.  
Let's say you have placed placeholder.sdd and placeholder.vtree in the aforementioned directories, as we
explaiend before, a class named SemanticLoss_placeholder will be created; this means
that a class named VNU_placeholder will also be created, and you will be able to log the VNU stats in tensorboard
by simply adding it to the generator statistics:  
```bash
    "GENERATOR_STATISTICS": ["VNU_placeholder"]
```
Again, you are still free to implement your own VNU class.


