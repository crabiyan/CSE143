This code was primarily developed and tested on the UCSC unix timeshare. It is also confirmed to work on a mac machine.

How to run this code:

    Prerequisites:
        -Have all 3 data text files (training, dev, test) located in a specific directory and named as follows:
            -training file: "1b_benchmark.train.tokens"
            -dev file: "1b_benchmark.dev.tokens"
            -test file: "1b_benchmark.test.tokens"
        -Have Python3 installed on your system with the following packages installed (these should be installed by default with Python3):
            -sys - System-specific parameters and functions
            -math - Mathematical functions
            -fractions - Rational numbers

    Running:
        -'python3 A1.py {path}'
        -where path represents the path to the directory containing all the data files
        -ex: 'python3 A1.py .'
        -The program will create 3 new files in the directory of A1.py