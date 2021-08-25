
class Predict(object):

    def forecast(self, df=None, val=None, lvl=0, hrchy=[]):
        if lvl == 0:
            temp = []
            for key, val in df.items():
                hrchy.append(key)
                result = self.forecast(val=val, lvl=lvl+1, hrchy=hrchy)
                temp.extend(result)
                hrchy.remove(key)

        elif lvl < self.hrchy_level:
            temp = []
            for key_hrchy, val_hrchy in val.items():
                hrchy.append(key_hrchy)
                result = self.forecast(val=val_hrchy, lvl=lvl+1, hrchy=hrchy)
                temp.extend(result)
                hrchy.remove(key_hrchy)

            return temp

        elif lvl == self.hrchy_level:
            for key_hrchy, val_hrchy in val.items():
                if len(val_hrchy) > self.n_test:
                    hrchy.append(key_hrchy)
                    temp = []
                    for algorithm in self.model_univ_list:
                        prediction = self.model_univ[algorithm](history=val_hrchy[self.target_feature].to_numpy(),
                                                                cfg=self.cfg_dict[algorithm],
                                                                pred_step=self.n_test)
                        temp.append(hrchy + [algorithm, prediction])
                    hrchy.remove(key_hrchy)

            return temp

        return temp