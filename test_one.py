def Sum(A,B):
    sum = A+B
    return sum

def test_Sum():
    assert (5+4) == Sum(5,4)

def test_noSum():
    assert (5+5) != Sum(5,4)