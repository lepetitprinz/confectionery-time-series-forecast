from baseline.analysis.SalesAnalysis import SalesAnalysis

# Execute Configuration
step_cfg = {
    'cls_load': False,
    'cls_prep': True,
    'cls_view': True
}

data_cfg = {
    'division': 'SELL_OUT',
    'item_lvl': 3,
    'cycle_yn': False,
    'date': {
        'from': '20190107',
        'to': '20220102'
    }
}

sa = SalesAnalysis(step_cfg=step_cfg, data_cfg=data_cfg)
sa.run()
