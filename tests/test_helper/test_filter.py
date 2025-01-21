""" Test suite for the filter funtionalities """
import pytest
import pandas as pd
import typeguard

manager = typeguard.install_import_hook("lit_ecology_classifier.helpers.filter")
import lit_ecology_classifier.helpers.filter as filterFunctionalities

# test the prepare_versions_to_filter method ----------------------------------------------------

# preare test matrix with modified input variables and expexcted output for each modification
@pytest.mark.parametrize(
    ("version_input", "expected_output"),
    [
    
        (["1"], pd.DataFrame({'image': ['a', 'b']}))
        (["2"], pd.DataFrame({'image': ['b', 'd']})),
        (["1", "2"], pd.DataFrame({'image': ['a', 'b', 'c', 'd']})),
    ],
)
def test_prepare_versions_to_filter(version_input, expected_output):

    # define same input dataframe for all test
    input_df = pd.DataFrame(
                    {
                     'image': ['a', 'b', 'c', 'd'],
                     'version_1': [True, True, False, False], 
                     'version_2': [True, False, True, False]
                     }
                    )
    x =filterFunctionalities.filter_versions(input_df, version_input)
    print(x)
    pd.testing.assert_frame_equal(x, expected_output)


# test the prepare_versions_to_filter method ----------------------------------------------------

# preare test matrix with modified input variables and expexcted output for each modification
@pytest.mark.parametrize(
        ("version_input", "expected_output"),
        [
            ("v1", ["v1"]),
            (["1", "2"], ["1", "2"]),
            ("all", []),
            (None, []),
        ],
    )

def test_prepare_versions_to_filter_all(version_input, expected_output):
    assert filterFunctionalities.prepare_args_to_filter(version_input) == expected_output

                                                        
manager.uninstall()