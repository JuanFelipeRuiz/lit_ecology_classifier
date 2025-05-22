import pytest

import lit_ecology_classifier.helpers.helpers as helpers


# Test the create_priority_classes ---------------------------------------------------------------

# preare test matrix with modified input variables and expexcted output for each modification
@pytest.mark.parametrize(
    ("rest_classes", "priority_classes", "expected_output"),
    [
        # return original class map if no rest classes are given
        ([], [] , {"class_a": 0, "class_b": 1, "class_c": 2}),

        # set class 1 as priority class and keep the rest classes
        ([], ["class_a"], {"class_a": 1, "rest": 0}),
        
        # keep only class 1 and 2 with class values since they are priority classes
        # and remove class 3
        (["class_a", "class_b"], [] , {"class_a": 0, "class_b": 1}),

        # keep class class 3 with original value and keep class 2 
        (["class_b"], ["class_c"] , {"rest": 0, "class_c": 1})
    ]
)


def test_rest_class_filter(expected_output, priority_classes, rest_classes):
    
    input_class_map = {"class_a": 1, "class_b": 2, "class_c": 3}
    
    assert helpers.filter_class_mapping( 
                            class_map= input_class_map,
                            rest_classes= rest_classes,
                            priority_classes= priority_classes) == expected_output
                                                         