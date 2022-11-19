

diagnosis2idx = {
        'pigment_network': {
            'absent': 0,
            'typical': 1,
            'atypical': 2,
            },
        'streaks': {
            'absent': 0,
            'irregular': 1,
            'regular': 2,
            },
        'pigmentation': {
            'localized irregular': 0,
            'localized regular': 1,
            'diffuse regular': 2,
            'absent': 3,
            'diffuse irregular': 4,
            },
        'regression_structures': {
            'absent': 0,
            'combinations': 1,
            'white areas': 2,
            'blue areas': 3,
            },
        'dots_and_globules': {
            'absent': 0,
            'irregular': 1,
            'regular': 2,
            },
        'blue_whitish_veil': {
            'absent': 0,
            'present': 1,
            },
        'vascular_structures': {
            'arborizing': 0,
            'within regression': 1,
            'wreath': 2,
            'absent': 3,
            'dotted': 4,
            'hairpin': 5,
            'comma': 6,
            'linear irregular': 7,
            },
        }

is_positive = {
        'pigment_network': { 2 },
        'streaks': { 1 },
        'pigmentation': { 0, 4 },
        'regression_structures': { 1, 2, 3 },
        'dots_and_globules': { 1 },
        'blue_whitish_veil': { 1 },
        'vascular_structures': { 4, 7 },
        }

assert diagnosis2idx.keys() == is_positive.keys()







# vim: set sw=4 sts=4 expandtab :
