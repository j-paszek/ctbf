import pytest

from evaluator import grf

test_cases = [
    # \phi \psi; missing label impact
    ("(a,b)c;", "(a,b);"),
    # 1,2,3; true tree, improved reconstruction (f internal), standard NJ
    ("((a,b)f,(c,d)e)g;", "((a,b)f,c);"),
    ("((a,b)f,(c,d)e)g;", "(((a,b),f),c);"),
    # I,II,III,IV; true tree, ideal only label missing, standard, wrong
    ("((a,b)c,(d)d)c;", "((a,b),(d)d);"),
    ("((a,b)c,(d)d)c;", "((a,b),d);"),
    ("((a,b)c,(d)d)c;", "(a,b)d;"),
    # 1a, 2,3; extended true tree
    ("((((a,x)y,z)v,b)f,(c,d)e)g;", "((a,b)f,c);"),
    ("((((a,x)y,z)v,b)f,(c,d)e)g;", "(((a,b),f),c);"),
    # 1,2,3 clearing labels
    ("((a,b)f,(c,d));", "((a,b)f,c);"),
    ("((a,b)f,(c,d));", "(((a,b),f),c);"),
    # test idea1
    ("((a,b)c,(d)d)c;", "((a,b),c);"),
    ("((a,b)c,(d)d)c;", "((a,b),e);")
]


@pytest.mark.parametrize("tree_A,tree_B", test_cases)
def test_grf(tree_A, tree_B):
    print(grf(tree_A, tree_B))

# METRIC UPDATE IDEA - replace nodes with blank
# for an internal node label
# if it is a leaf in treeA or treeB it stays
# if it is an internal node in both treeA and treeB it stays
#
# @pytest.mark.parametrize("tree_A,tree_B", test_cases)
# def test_grf_idea1(tree_A, tree_B):
#     mod_1, mod_2 = idea1(tree_A, tree_B)
#     print(mod_1, mod_2)
#     print(grf(mod_1, mod_2)