
import pandas as pd
import pytest

import lit_ecology_classifier.helpers.filter as filterFunctionalities


# test the prepare_versions_to_filter method ----------------------------------------------------

# preare test matrix with modified input variables and expexcted output for each modification
@pytest.mark.parametrize(
    ("version_input", "expected_output"),
    [
        (["1"], pd.DataFrame({'image': ['a', 'b']})),
        (["2"], pd.DataFrame({'image': ['b', 'd']})),
        (["1", "2"], pd.DataFrame({'image': ['a', 'b', 'c', 'd']})),
    ],
)
def test_prepare_versions_to_filter(version_input, expected_output):

    # define same input dataframe for all test
    input_df = pd.DataFrame(
                    {
                     'image': ['a', 'b', 'c', 'd'],
                     'version_v1': [True, True, False, False], 
                     'version_v2': [True, False, True, False]
                     }
                    )
    
    pd.testing.assert_frame_equal(filterFunctionalities.filter_versions(version_input, input_df), expected_output)


# test the prepare_versions_to_filter method ----------------------------------------------------

# preare test matrix with modified input variables and expexcted output for each modification
@pytest.mark.parametrize(
        ("version_input", "expected_output"),
        [
            ("1", ["1"]),
            (["1", "2"], ["1", "2"]),
            ("all", []),
            (None, []),
        ],
    )

def test_prepare_versions_to_filter_all(version_input, expected_output):
    assert filterFunctionalities.prepare_versions_to_filter(version_input) == expected_output

# Test the create_priority_classes ---------------------------------------------------------------

# preare test matrix with modified input variables and expexcted output for each modification
@pytest.mark.parametrize(
    ("priority_classes", "expected_output"),
    [
        (["class_1"], {"class_1": 1, "class_2": 0, "class_3": 0}),
        (["class_1", "class_2"], {"class_1": 1, "class_2": 2, "class_3": 0}),
    ],
)

def test_define_priority_classes(priority_classes, expected_output):
    input_class_map = {"class_1": 1, "class_2": 2, "class_3": 3}
    
    assert filterFunctionalities.create_priority_classes(priority_classes, input_class_map) == expected_output

# test rest_class_filtering -------------------------------------------------------------------------

# preare test matrix with modified input variables and expexcted output for each modification
@pytest.mark.parametrize(
    ("rest_classes", "priority_classes", "expected_output"),
    [
        ([], [] , {"class_1": 1, "class_2": 2, "class_3": 3}),

        (["class_1", "class_2"], [] , {"class_1": 1, "class_2": 2}),

        (["class_1", "class_2"], ["class_3"] , {"class_1": 1, "class_2": 2, "class_3": 3}),

    ]
)


def test_rest_class_filter(expected_output, priority_classes, rest_classes):
    
    input_class_map = {"class_1": 1, "class_2": 2, "class_3": 3}
    
    assert filterFunctionalities.class_filter( 
                            class_map= input_class_map,
                            rest_classes= rest_classes,
                            priority_classes= priority_classes) == expected_output
                                                         
                                                        
    