from baseline.analysis.PredCompare import PredCompare

data_cfg = {
    'division': 'SELL_OUT',
    'item_lvl': 3,
    'cycle_yn': False,
    'date': {
        'from': '20211227',
        'to': '20220102'
    },
    'data_vrsn_cd': '20201228-20211226'
}

# Initialize class
comp = PredCompare(data_cfg=data_cfg)

# run
comp.run()
