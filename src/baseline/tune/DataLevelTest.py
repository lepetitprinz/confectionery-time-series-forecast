from baseline.deployment.Pipleline import Pipeline
from typing import List


class DataLevelTest(object):
    def __init__(self, division: str, cust_lvl_list: List[int], prod_lvl_list: List[int],
                 save_step_yn=False, load_step_yn=False, save_db_yn=False):
        # Data Configuration
        self.division = division

        # Data Level Test Configuration
        self.cust_lvl_list = cust_lvl_list
        self.prod_lvl_list = prod_lvl_list

        # Save & Load Configuration
        self.save_steps_yn = save_step_yn
        self.load_step_yn = load_step_yn
        self.save_db_yn = save_db_yn

    def test(self):
        for cust_lvl in self.cust_lvl_list:
            for prod_lvl in self.prod_lvl_list:
                pipeline = Pipeline(division=self.division, cust_lvl=cust_lvl, prod_lvl=prod_lvl,
                                    save_step_yn=self.save_steps_yn, load_step_yn=self.load_step_yn,
                                    save_db_yn=self.save_db_yn)
                pipeline.run()
