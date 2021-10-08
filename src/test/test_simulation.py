from simulation.deployment.Pipeline import Pipeline

division = 'SELL_IN'
hrchy_lvl = 4
lag = 'w1'

save_step_yn = True
load_step_yn = True
save_db_yn = False

pipeline = Pipeline(division=division, hrchy_lvl=hrchy_lvl, lag=lag,
                    save_step_yn=save_step_yn, load_step_yn=load_step_yn)
pipeline.run()
