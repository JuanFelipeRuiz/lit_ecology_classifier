import hashlib

import pandas as pd


from lit_ecology_classifier.splitting.split import HashGenerator

def test_sha256_from_list():
    data = ['a', 'b', 'c']
    expected_hash = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert HashGenerator.sha256_from_list(data) == expected_hash 


def test_generate_hash_dict_from_split():
    split = pd.DataFrame({'split': ['train', 'train', 'train','test', 'test'] ,
                           'hash_value' : ['a', 'b', 'c', 'd', 'e' ] })
    expected_hash = {
        'train': 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
        'test': '959a45d44e6fcf58361ed004681556fe50129f2109e817dec098c00c9e5d2578'
    }
    assert HashGenerator.generate_hash_dict_from_split(split, hash_col_name= 'hash_value') == expected_hash


