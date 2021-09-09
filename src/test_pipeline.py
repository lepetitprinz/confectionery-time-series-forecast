from baseline.deployment.Pipleline import Pipeline

division = 'SELL-IN'
cust_lvl = 0
prod_lvl = 3
save_step_yn = True
load_step_yn = True

pipeline = Pipeline(division=division,
                    cust_lvl=cust_lvl,
                    prod_lvl=prod_lvl,
                    save_step_yn=save_step_yn,
                    load_step_yn=load_step_yn)

pipeline.run()
