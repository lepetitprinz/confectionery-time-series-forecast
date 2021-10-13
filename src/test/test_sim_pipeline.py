from simulation.deployment.Pipeline import Pipeline

division = 'sell_in'
hrchy_lvl = 4
lag = 'w1'

save_obj_yn = True
load_obj_yn = True
save_db_yn = False
grid_search_yn = False

pipeline = Pipeline(division=division, hrchy_lvl=hrchy_lvl, lag=lag,
                    save_obj_yn=save_obj_yn, load_obj_yn=load_obj_yn, save_db_yn=save_db_yn)
pipeline.run()
