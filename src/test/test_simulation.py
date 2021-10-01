from simulation.deployment.Pipeline import Pipeline

division = 'SELL_IN'
hrchy_lvl = 4
lag = 'w1'

pipeline = Pipeline(division=division, hrchy_lvl=hrchy_lvl, lag=lag,
                    load_step_yn=True)
pipeline.run()
