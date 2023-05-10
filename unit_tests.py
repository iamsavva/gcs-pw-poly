from util import YAY, INFO
from bad_LQR_unit_test import make_a_simple_bad_lqr_test
from good_LQR_unit_test import make_a_simple_lqr_test
from chain_unit_tests import make_simple_chain_point_to_point_test

if __name__ == "__main__":
    make_simple_chain_point_to_point_test()
    make_a_simple_bad_lqr_test()
    make_a_simple_lqr_test()
    
    YAY("---------------")
    YAY("All tests passed!")
    INFO("If you broke the code, I wouldn't know")
