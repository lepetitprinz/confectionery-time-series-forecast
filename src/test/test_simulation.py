from simulation.deployment.Pipeline import Pipeline

division = 'SELL_IN'

pipeline = Pipeline(division=division, load_step_yn=True)
pipeline.run()
