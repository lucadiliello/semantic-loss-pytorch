import unittest
import re as regex
import sys
import random
import os
import numpy as np
from sympy import *

# add parent directory so we can import the module
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from constraints_to_cnf import ConstraintsToCnf
from sympy.logic.utilities.dimacs import load_file
from sympy.parsing.sympy_parser import *

"""
DISCLAIMER: these tests were written kinda fast and are without
comments, and with bad variable naming, however each one is quite short and
should be easy to understand what's happening.

TODO: operators that go recursively when doing test_output_equivalence_iterations
TODO: operators using N > 2 arguments when doing test_output_equivalence_iterations
"""


def cleanup(file):
    os.system("rm %s" % file)


def to_file_in_tmp(string, caller):
    # get function name
    fname = str(caller).split("test")[1].split()[0]
    filename = "/tmp/%s.txt" % fname
    with open(filename, "w") as output:
        output.write(string)
    return filename


class TestReadData(unittest.TestCase):

    def test_noshape(self):
        string = "X1 >> X2"
        fname = to_file_in_tmp(string, self.test_noshape)

        with self.assertRaises(AssertionError):
            ConstraintsToCnf._read_data(fname)

        cleanup(fname)

    def test_badshape1(self):
        string = "shape 4,4"
        fname = to_file_in_tmp(string, self.test_badshape1)

        with self.assertRaises(AssertionError):
            ConstraintsToCnf._read_data(fname)

        cleanup(fname)

    def test_badshape2(self):
        string = "shape [4 4]"
        fname = to_file_in_tmp(string, self.test_badshape2)

        with self.assertRaises(AssertionError):
            ConstraintsToCnf._read_data(fname)

        cleanup(fname)

    def test_badshape3(self):
        string = "shape [4; 4]"
        fname = to_file_in_tmp(string, self.test_badshape3)

        with self.assertRaises(AssertionError):
            ConstraintsToCnf._read_data(fname)

        cleanup(fname)

    def test_negativeindexshape(self):
        string = "shape [4, -4]"
        fname = to_file_in_tmp(string, self.test_negativeindexshape)

        with self.assertRaises(AssertionError):
            ConstraintsToCnf._read_data(fname)

        cleanup(fname)

    def test_noconstraints(self):
        string = "shape [4, 4]"
        fname = to_file_in_tmp(string, self.test_noconstraints)

        # we expect this to pass
        ConstraintsToCnf._read_data(fname)

        cleanup(fname)

    def test_hasconstraints(self):
        string = "shape [4, 4]\n"
        string += "X1 >> X2"
        fname = to_file_in_tmp(string, self.test_hasconstraints)

        # we expect this to pass
        ConstraintsToCnf._read_data(fname)

        cleanup(fname)

    def test_check_correct_result1(self):
        string = "shape [4, 4]\n"
        string += "X1 >> X2\n"
        fname = to_file_in_tmp(string, self.test_check_correct_result1)

        # we expect this line to pass
        shape, constr = ConstraintsToCnf._read_data(fname)
        assert shape == [4, 4]
        assert type(constr) == list and len(constr) == 1
        assert constr[0] == "X1 >> X2"

        cleanup(fname)

    def test_check_correct_result2(self):
        string = "shape [4, 4]\n"
        string += "X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "X1 >> X2\n"

        fname = to_file_in_tmp(string, self.test_check_correct_result2)

        # we expect this line to pass
        shape, constr = ConstraintsToCnf._read_data(fname)
        assert shape == [4, 4]
        assert type(constr) == list and len(constr) == 5
        for c in constr:
            assert c == "X1 >> X2"

        cleanup(fname)

    def test_check_correct_result3(self):
        string = "# X1 >> X2\n"
        string += "shape [4, 4]\n"
        string += "# X1 >> X2\n"
        string += "# X1 >> X2\n"
        string += "# X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "# X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "# X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "# X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "# X1 >> X2\n"
        string += "# X1 >> X2\n"
        fname = to_file_in_tmp(string, self.test_check_correct_result3)

        # we expect this line to pass
        shape, constr = ConstraintsToCnf._read_data(fname)
        assert shape == [4, 4]
        assert type(constr) == list and len(constr) == 5
        for c in constr:
            assert c == "X1 >> X2"

        cleanup(fname)

    def test_check_correct_result4(self):
        string = "# X1 >> X2\n"
        string += "shape [4, 4]\n"
        string += "# X1 >> X2\n"
        string += "# X1 >> X2\n"
        string += "# X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "# X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "# X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "# X1 >> X2\n"
        string += "X1 >> X2\n"
        string += "# X1 >> X2\n"
        string += "# X1 >> X2\n"
        fname = to_file_in_tmp(string, self.test_check_correct_result4)

        # we expect this line to pass
        shape, constr = ConstraintsToCnf._read_data(fname)
        assert shape == [4, 4]
        assert type(constr) == list and len(constr) == 5
        for c in constr:
            assert c == "X1 >> X2"

        cleanup(fname)


valid = ConstraintsToCnf._assert_is_valid_variable
# default stuff for easy access
dfshape = [4, 10, 15]
dfstride = [150, 15, 1]
dftotal = 150 * 15


class TestValidVar(unittest.TestCase):

    def test_incorrect_syntax1(self):
        with self.assertRaises(AssertionError):
            valid("X_1", dfshape, dfstride, dftotal)

    def test_incorrect_syntax2(self):
        with self.assertRaises(AssertionError):
            valid("Y_1", dfshape, dfstride, dftotal)

    def test_incorrect_syntax3(self):
        with self.assertRaises(AssertionError):
            valid("X_1_", dfshape, dfstride, dftotal)

    def test_incorrect_syntax4(self):
        with self.assertRaises(AssertionError):
            valid("X_1", dfshape, dfstride, dftotal)

    def test_incorrect_syntax5(self):
        with self.assertRaises(AssertionError):
            valid("X1X", dfshape, dfstride, dftotal)

    def test_incorrect_syntax6(self):
        with self.assertRaises(AssertionError):
            valid("X1X", dfshape, dfstride, dftotal)

    def test_incorrect_syntax7(self):
        with self.assertRaises(AssertionError):
            valid("X_", dfshape, dfstride, dftotal)

    def test_incorrect_syntax8(self):
        with self.assertRaises(AssertionError):
            valid("(X_1)", dfshape, dfstride, dftotal)

    def test_incorrect_syntax9(self):
        with self.assertRaises(AssertionError):
            valid("^X_1$", dfshape, dfstride, dftotal)

    def test_incorrect_syntax10(self):
        with self.assertRaises(AssertionError):
            valid("1X_1", dfshape, dfstride, dftotal)

    def test_incorrect_syntax10(self):
        with self.assertRaises(AssertionError):
            valid("1X_1", dfshape, dfstride, dftotal)

    def test_incorrect_syntax11(self):
        with self.assertRaises(AssertionError):
            valid("X_-1", dfshape, dfstride, dftotal)

    def test_incorrect_syntax12(self):
        with self.assertRaises(AssertionError):
            valid("X_1.10", dfshape, dfstride, dftotal)

    def test_correct_syntax1(self):
        # expect to pass
        valid("X_1_1_1", dfshape, dfstride, dftotal)

    def test_correct_syntax2(self):
        # expect to pass
        valid("X_3_9_14", dfshape, dfstride, dftotal)

    def test_correct_syntax3(self):
        # expect to pass
        valid("X_0_0_0", dfshape, dfstride, dftotal)

    def test_not_enough_indexes(self):
        with self.assertRaises(AssertionError):
            valid("X_0_0", dfshape, dfstride, dftotal)

    def test_too_many_indexes(self):
        with self.assertRaises(AssertionError):
            valid("X_0_0_10_10", dfshape, dfstride, dftotal)

    def test_out_of_bound_index1(self):
        with self.assertRaises(AssertionError):
            valid("X_4_2_3", dfshape, dfstride, dftotal)

    def test_out_of_bound_index2(self):
        with self.assertRaises(AssertionError):
            valid("X_2_10_3", dfshape, dfstride, dftotal)

    def test_out_of_bound_index3(self):
        with self.assertRaises(AssertionError):
            valid("X_2_9_15", dfshape, dfstride, dftotal)

    def test_out_of_bound_index4(self):
        with self.assertRaises(AssertionError):
            valid("X_200_900_1500", dfshape, dfstride, dftotal)


toDIMACS = ConstraintsToCnf.expression_to_cnf


class TestOutput(unittest.TestCase):

    def test_output_only_shape(self):
        string = "#X1.2\n"
        string += "shape [4,4]\n"
        string += "#X1.2\n"
        inputfilepath = to_file_in_tmp(string, self.test_output_only_shape)
        outputfilepath = "/tmp/t1.txt"

        toDIMACS(inputfilepath, outputfilepath)

        with open(outputfilepath, "r") as outputfile:
            string = [s for s in outputfile.read().split("\n") if s != "" and s[0] != "c"]
            self.assertEqual(string[0], "p cnf 16 1")
            self.assertEqual(string[1], "True 0")

        cleanup(inputfilepath)
        cleanup(outputfilepath)

    def test_output_1var(self):
        string = "#X1.2\n"
        string += "shape [7,8]\n"
        string += "X1.7\n"
        string += "#X1.2\n"
        inputfilepath = to_file_in_tmp(string, self.test_output_1var)
        outputfilepath = "/tmp/t2.txt"

        toDIMACS(inputfilepath, outputfilepath)

        with open(outputfilepath, "r") as outputfile:
            string = [s for s in outputfile.read().split("\n") if s != "" and s[0] != "c"]
            self.assertEqual(string[0], "p cnf 56 1")
            self.assertEqual(string[1], "16 0")

        cleanup(inputfilepath)
        cleanup(outputfilepath)

    def test_output_2var(self):
        string = "shape [1,1,1,1,1,1,1,1]\n"
        string += "X0.0.0.0.0.0.0.0"
        inputfilepath = to_file_in_tmp(string, self.test_output_2var)
        outputfilepath = "/tmp/t3.txt"

        toDIMACS(inputfilepath, outputfilepath)

        with open(outputfilepath, "r") as outputfile:
            string = [s for s in outputfile.read().split("\n") if s != "" and s[0] != "c"]
            self.assertEqual(string[0], "p cnf 1 1")
            self.assertEqual(string[1], "1 0")

        cleanup(inputfilepath)
        cleanup(outputfilepath)

    def test_output_2var_repeated(self):
        string = "shape [1,1,1,1,1,1,1,1]\n"
        string += "X0.0.0.0.0.0.0.0\n"
        string += "X0.0.0.0.0.0.0.0\n"
        string += "X0.0.0.0.0.0.0.0\n"
        string += "#X0.0.0.0.0.0.0.0\n"
        string += "X0.0.0.0.0.0.0.0\n"
        string += "#X0.0.0.0.0.0.0.0\n"
        string += "X0.0.0.0.0.0.0.0\n"
        string += "#X0.0.0.0.0.0.0.0\n"
        string += "X0.0.0.0.0.0.0.0\n"
        inputfilepath = to_file_in_tmp(string, self.test_output_2var_repeated)
        outputfilepath = "/tmp/t2.txt"

        toDIMACS(inputfilepath, outputfilepath)

        with open(outputfilepath, "r") as outputfile:
            string = [s for s in outputfile.read().split("\n") if s != "" and s[0] != "c"]
            self.assertEqual(string[0], "p cnf 1 1")
            self.assertEqual(string[1], "1 0")

        cleanup(inputfilepath)
        cleanup(outputfilepath)

    def test_output_nvar1(self):
        string = "shape [5,6]\n"
        for i in range(5):
            for u in range(6):
                string += "X.%s.%s\n" % (i, u)
        inputfilepath = to_file_in_tmp(string, self.test_output_nvar1)
        outputfilepath = "/tmp/t4.txt"

        toDIMACS(inputfilepath, outputfilepath)

        seen_vars = set()
        with open(outputfilepath, "r") as outputfile:
            string = [s for s in outputfile.read().split("\n") if s != "" and s[0] != "c"]
            self.assertEqual(string[0], "p cnf 30 30")

            [seen_vars.add(s.split()[0]) for s in string[1:]]
            for i in range(1, 31):
                self.assertIn(str(i), seen_vars)

        cleanup(inputfilepath)
        cleanup(outputfilepath)

    def test_output_nvar2(self):
        tshape = np.array([5, 6, 50])
        total = np.cumprod(tshape)[-1]
        string = "shape [5,6,50]\n"
        for i in range(tshape[0]):
            for u in range(tshape[1]):
                for z in range(tshape[2]):
                    string += "X.%s.%s.%s\n" % (i, u, z)
        inputfilepath = to_file_in_tmp(string, self.test_output_nvar2)
        outputfilepath = "/tmp/t5.txt"

        toDIMACS(inputfilepath, outputfilepath, 8)

        seen_vars = set()
        with open(outputfilepath, "r") as outputfile:
            string = [s for s in outputfile.read().split("\n") if s != "" and s[0] != "c"]
            self.assertEqual(string[0], "p cnf %s %s" % (total, total))

            [seen_vars.add(s.split()[0]) for s in string[1:]]
            for i in range(1, total):
                self.assertIn(str(i), seen_vars)

        cleanup(inputfilepath)
        cleanup(outputfilepath)

    def test_output_nvar3(self):
        tshape = np.array([4, 4, 4])
        total = np.cumprod(tshape)[-1]
        string = "shape [4,4,4]\n"
        for i in range(tshape[0]):
            for u in range(tshape[1]):
                for z in range(tshape[2]):
                    string += "X.%s.%s.%s|" % (i, u, z)
        string = string[:-1] + "\n"
        inputfilepath = to_file_in_tmp(string, self.test_output_nvar3)
        outputfilepath = "/tmp/t6.txt"

        toDIMACS(inputfilepath, outputfilepath, 8)

        seen_vars = set()
        with open(outputfilepath, "r") as outputfile:
            string = [s for s in outputfile.read().split("\n") if s != "" and s[0] != "c"]
            self.assertEqual(len(string), 2)
            self.assertEqual(string[0], "p cnf %s %s" % (total, 1))

            [seen_vars.add(s) for s in string[1].split()]
            for i in range(1, total):
                self.assertIn(str(i), seen_vars)

        cleanup(inputfilepath)
        cleanup(outputfilepath)

    def test_output_andNors1(self):
        tshape = np.array([20])
        total = np.cumprod(tshape)[-1]

        string = "shape [20]\n"
        for i in range(0, 20, 2):
            string += "X%s | X%s\n" % (i, i + 1)

        inputfilepath = to_file_in_tmp(string, self.test_output_nvar3)
        outputfilepath = "/tmp/t7.txt"

        toDIMACS(inputfilepath, outputfilepath)

        with open(outputfilepath, "r") as outputfile:
            string = [s for s in outputfile.read().split("\n") if s != "" and s[0] != "c"]
            self.assertEqual(len(string), 11)
            self.assertEqual(string[0], "p cnf %s %s" % (total, 10))

        cleanup(inputfilepath)
        cleanup(outputfilepath)

    def test_output_equivalence1(self):
        tshape = np.array([20])

        string = "shape [20]\n"
        prepareinput = ""
        for i in range(0, 20, 2):
            prepareinput += "(cnf_%s | cnf_%s) &" % (i + 1, i + 2)
            string += "(X%s | X%s)\n" % (i, i + 1)
        prepareinput = prepareinput[:-1]

        inputfilepath = to_file_in_tmp(string, self.test_output_nvar3)
        outputfilepath = "/tmp/t8.txt"

        toDIMACS(inputfilepath, outputfilepath)

        output_parsed_by_sympy = load_file(outputfilepath)
        input_parsed_by_sympy = parse_expr(prepareinput)
        self.assertTrue(input_parsed_by_sympy.equals(output_parsed_by_sympy))
        self.assertTrue(to_cnf(input_parsed_by_sympy).equals(output_parsed_by_sympy))

        cleanup(inputfilepath)
        cleanup(outputfilepath)

    operators = [("%s | %s", 2), ("%s & %s", 2), ("Xor(%s, %s)", 2), ("Nand(%s, %s)", 2), ("Nor(%s, %s)", 2),
                 ("ITE(%s, %s ,%s)", 3), ("%s >> %s", 2), ("%s << %s", 2), ("Implies(%s, %s)", 2),
                 ("Equivalent(%s, %s)", 2)]

    def test_output_equivalence_iterations(self):
        iterations = 20
        nconstraints = 100

        for _ in range(iterations):
            tshape = np.array([400])

            string = "shape [400]\n"
            prepareinput = ""
            for _ in range(0, nconstraints, 1):
                operator, requiredargs = self.operators[random.randrange(0, len(self.operators))]
                prepareinputargs = []
                stringargs = []

                # need to do this to check Xor not having the same exact arguments or it evaluates to False
                vars = [random.randrange(0, tshape[0]) for _ in range(requiredargs)]
                if "Xor" in operator and (len(set(vars)) != requiredargs):
                    # duplicate vars and xor operator
                    continue

                for var in vars:
                    if random.randrange(0, 2):
                        prepareinputargs.append("cnf_%s" % str(var + 1))
                        stringargs.append("X%s" % var)
                    else:
                        prepareinputargs.append("~cnf_%s" % str(var + 1))
                        stringargs.append("~X%s" % var)

                prepareinput += (" ( " + (operator % tuple(prepareinputargs)) + " ) &")
                string += ((operator % tuple(stringargs)) + "\n")

            prepareinput = prepareinput[:-1]

            inputfilepath = to_file_in_tmp(string, self.test_output_equivalence_iterations)
            outputfilepath = "/tmp/t9.txt"

            toDIMACS(inputfilepath, outputfilepath)

            output_parsed_by_sympy = load_file(outputfilepath)
            input_parsed_by_sympy = parse_expr(prepareinput, evaluate=False)
            assert to_cnf(input_parsed_by_sympy).equals(output_parsed_by_sympy), "%s \n %s" % (
                to_cnf(input_parsed_by_sympy), output_parsed_by_sympy)
            self.assertTrue(to_cnf(input_parsed_by_sympy).equals(output_parsed_by_sympy))

            cleanup(inputfilepath)
            cleanup(outputfilepath)

    def test_output_equivalence_iterations_multiprocess(self):
        iterations = 20
        nconstraints = 100

        for _ in range(iterations):
            tshape = np.array([400])

            string = "shape [400]\n"
            prepareinput = ""
            for _ in range(0, nconstraints, 1):
                operator, requiredargs = self.operators[random.randrange(0, len(self.operators))]
                prepareinputargs = []
                stringargs = []

                # need to do this to check Xor not having the same exact arguments or it evaluates to False
                vars = [random.randrange(0, tshape[0]) for _ in range(requiredargs)]
                if "Xor" in operator and (len(set(vars)) != requiredargs):
                    # duplicate vars and xor operator
                    continue

                for var in vars:
                    if random.randrange(0, 2):
                        prepareinputargs.append("cnf_%s" % str(var + 1))
                        stringargs.append("X%s" % var)
                    else:
                        prepareinputargs.append("~cnf_%s" % str(var + 1))
                        stringargs.append("~X%s" % var)

                prepareinput += (" ( " + (operator % tuple(prepareinputargs)) + " ) &")
                string += ((operator % tuple(stringargs)) + "\n")

            prepareinput = prepareinput[:-1]

            inputfilepath = to_file_in_tmp(string, self.test_output_equivalence_iterations)
            outputfilepath = "/tmp/t10.txt"

            toDIMACS(inputfilepath, outputfilepath, nprocesses=random.randrange(1, 30))

            output_parsed_by_sympy = load_file(outputfilepath)
            input_parsed_by_sympy = parse_expr(prepareinput, evaluate=False)
            self.assertTrue(to_cnf(input_parsed_by_sympy).equals(output_parsed_by_sympy))

            cleanup(inputfilepath)
            cleanup(outputfilepath)


if __name__ == '__main__':
    unittest.main()
    np.random.seed(1337)
    random.seed(1337)
