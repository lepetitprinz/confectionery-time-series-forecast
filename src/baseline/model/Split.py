import common.util as util


class Split(object):
    def __init__(self, division, hrchy:list):
        self.division = division
        # self.cust_lvl = cust_lvl
        # self.prod_lvl = prod_lvl

        # Data Level Configuration
        self.hrchy = hrchy
        self.hrchy_level = len(hrchy) - 1

    def split(self, data):
        split_rate = self.hrchy_recursion_with_key(hrchy_lvl=self.hrchy_level,
                                                   fn=self.split_data,
                                                   df=data)

    def split_data(self, hrchy, df):
        return None

    def hrchy_recursion_with_key(self, hrchy_lvl, fn=None, df=None, val=None, lvl=0, hrchy=[]):
        if lvl == 0:
            temp = []
            for key, val in df.items():
                hrchy.append(key)
                result = self.hrchy_recursion_with_key(hrchy_lvl=hrchy_lvl, fn=fn, val=val,
                                                  lvl=lvl + 1, hrchy=hrchy)
                temp.extend(result)
                hrchy.remove(key)

        elif lvl < hrchy_lvl:
            temp = []
            for key_hrchy, val_hrchy in val.items():
                hrchy.append(key_hrchy)
                result = self.hrchy_recursion_with_key(hrchy_lvl=hrchy_lvl, fn=fn, val=val_hrchy,
                                                  lvl=lvl + 1, hrchy=hrchy)
                temp.extend(result)
                hrchy.remove(key_hrchy)

            return temp

        elif lvl == hrchy_lvl:
            temp = []
            for key_hrchy, val_hrchy in val.items():
                hrchy.append(key_hrchy)
                result = fn(hrchy, val_hrchy)
                temp.extend(result)
                hrchy.remove(key_hrchy)

            return temp

        return temp