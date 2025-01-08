"""
Script to split the data into training and testing sets. 
Can only use split or filter strategies that are predifined inside of
the lit_ecology_classifier package. To use custom strategies, follow the template   
provided in the filter or split_strategy base class and add the strategy to the
strategies directory. 
"""

import logging
import pathlib
import sys

from time import time

from lit_ecology_classifier.data_overview.overview_creator import OverviewCreator
from lit_ecology_classifier.data_overview.images_copier import ImageCopier
from lit_ecology_classifier.helpers.argparser import overview_argparser
