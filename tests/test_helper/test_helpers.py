import pytest

import lit_ecology_classifier.helpers.helpers as helpers


# Test the create_priority_classes ---------------------------------------------------------------

# preare test matrix with modified input variables and expexcted output for each modification
@pytest.mark.parametrize(
    ("rest_classes", "priority_classes", "expected_output"),
    [
        # return original class map if no rest classes are given
        ([], [] , {"class_1": 1, "class_2": 2, "class_3": 3}),

        # set class 1 as priority class and keep the rest classes
        ([], ["class_1"], {"class_1": 1, "rest": 0}),
        
        # keep only class 1 and 2 with class values since they are priority classes
        # and remove class 3
        (["class_1", "class_2"], [] , {"class_1": 1, "class_2": 2}),

        # keep class class 3 with original value and keep class 2 
        (["class_2"], ["class_3"] , {"rest": 0, "class_3": 3})
    ]
)


def test_rest_class_filter(expected_output, priority_classes, rest_classes):
    
    input_class_map = {"class_1": 1, "class_2": 2, "class_3": 3}
    
    assert helpers.filter_class_mapping( 
                            class_map= input_class_map,
                            rest_classes= rest_classes,
                            priority_classes= priority_classes) == expected_output
                                                         