These tests are about checking that the two versions
of pypsdd (python2 and the roughly ported version to python3) output
the same tensorflow graph given the same vtree and sdd.  
**Calling write_all_graphs.py with python2 will call the original/official
python2 version of pysdd**, and write all the graphs related
to the cnfs/vtrees/sdds you can find in the respective directories in the py2graphs directory.  
Similarly, **calling it with python3 will use the version of pysdd
which is present in this repository**.
Once graphs for both versions are created, you can run check_diffs.py, that will compare the two versions of all graphs.  

Note: when running the python2 version you have to have it installed, it is not provided in this directory. Also make sure you have
tensorflow.  Again, when running with python3, make sure to have tf and the python3 version of the library installed, a package
that can be used to install it can be found in the py3psdd directory, py3psdd_package.zip.

Note2: some vtrees fail to be read for both versions. Checking
why that is happening is on the todo list.
