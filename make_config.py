import json

config = {
    'ENV_LABEL': {
        #0 for cartpole env
        #1 for girdworld env
        'choices': (0,1),
    },
    0 : {
        'NEXT_ID': 0,
        'SHOW_WINDOWS': {
            'low': 0,
            'high': 1,
            'default': 0
        },
        'PARAM_NUMBER_OF_EPISODES': {
            'low': 1,
            'high': 1000,
            'default': 100
        },
        'PARAM_EPISODE_MAX_LENGTH': {
            'default': 1000
        },
        'PARAM_LEARNING_RATE': {
            'low': 0.01,
            'high': 0.9999,
            'default': 0.89
        },
        'PARAM_DISCOUNT_FACTOR': {
            'low': 0.01,
            'high': 0.99999,
            'default': 0.99
        },
        'PARAM_EPSILON': {
            'low': 0.001,
            'high': 0.99999,
            'default': 0.2
        },
        'TABLE_FILENAME' : ''
    },
    1 : {
        'NEXT_ID': 0,
        'SHOW_WINDOWS': {
            'low': 0,
            'high': 1,
            'default': 0
        },
        'PARAM_NUMBER_OF_EPISODES': {
            'low': 1,
            'high': 100,
            'default': 20
        },
        'PARAM_EPISODE_MAX_LENGTH': {
            'default': 1000
        },
        'PARAM_LEARNING_RATE': {
            'low': 0.5,
            'high': 0.9999,
            'default': 0.91
        },
        'PARAM_DISCOUNT_FACTOR': {
            'low': 0.2,
            'high': 0.99999,
            'default': 0.9
        },
        'PARAM_EPSILON': {
            'low': 0.001,
            'high': 0.99999,
            'default': 0.2
        },
        'TABLE_FILENAME' : ''
    }
}

with open('config.json', 'w') as f:
    json.dump(config, f)