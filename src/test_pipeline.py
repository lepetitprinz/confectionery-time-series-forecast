from baseline.deployment.Pipleline import Pipeline

division = 'SELL_IN'
cust_lvl = 0    # Customer_group - Customer
prod_lvl = 4    # Biz - Line - Brand - Item - SKU
save_step_yn = True
load_step_yn = True
save_db_yn = True

pipeline = Pipeline(division=division,
                    cust_lvl=cust_lvl,
                    prod_lvl=prod_lvl,
                    save_step_yn=save_step_yn,
                    load_step_yn=load_step_yn,
                    save_db_yn=save_db_yn)

pipeline.run()
