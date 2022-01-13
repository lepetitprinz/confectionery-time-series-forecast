from baseline.analysis.PredCompare import PredCompare

data_cfg = {
    'division': 'SELL_IN',
    'item_lvl': 3,
    'cycle_yn': False,
    'date': {
        'hist': {
            'from': '20210104',
            'to': '20220102'
        },
        'compare': {
            'from': '20220103',
            'to': '20220109'
        }
    },
    'data_vrsn_cd': '20210104-20220102'
}

# Initialize class
comp = PredCompare(data_cfg=data_cfg)

# run
comp.run()
