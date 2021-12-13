import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from recommend.deployment.Pipeline import Pipeline

item_col = 'sku_cd'
meta_col = 'bom_cd'

exec_cfg = {
    'save_step_yn': True,            # Save each step result to object or csv
    'save_db_yn': False,             #
}

pipeline = Pipeline(exec_cfg=exec_cfg, item_col=item_col, meta_col=meta_col)
# pipeline.run()
