

class Predict(object):
    def __init__(self):
        pass

    def _pred(self, pred_day: str, fitted_model: dict, applied_list: list):
        # Get season value and initial discount rate
        pred_datetime = dt.datetime(*list(map(int, pred_day.split('-'))))
        season = self.season_re[pred_datetime]
        init_disc = self.disc_last_week[pred_datetime]
        init_capa = self.capacity_re[pred_datetime]

        # Make initial values dataframe
        pred_input = self._get_pred_input(pred_day=pred_day, season=season, init_disc=init_disc,
                                          applied_list=applied_list)

        pred_result = self._pred_fitted_model(pred_datetime=pred_datetime, pred_input=pred_input,
                                              fitted_model=fitted_model)

        # Map lead time to prediction results
        lt_to_pred_result = self._get_lt_to_pred_result(pred_result=pred_result)

        result = self._map_rslt_to_lead_time(pred_final=lt_to_pred_result)

        # Result data convert to dataframe
        result_df = self._conv_to_dataframe(result=result, pred_datetime=pred_datetime,
                                            init_disc=init_disc, init_capa=init_capa)

        # Save the result dataframe
        self._save_result(result=result_df, pred_day=pred_day)

        print(f'Prediction result on {pred_day} is saved')