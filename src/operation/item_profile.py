import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from recommend.deployment.PipelineCycle import PipelineCycle

cfg = {
    'save_step_yn': False,            # Save each step result to object or csv
    'save_db_yn': True,             #
    'cycle': 'w'
}

print('------------------------------------------------')
print('Item Profiling')
print('------------------------------------------------')

# Check start time
print("Item Profiling Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

pipeline = PipelineCycle(cfg=cfg)
pipeline.run()

# Check end time
print("Item Profiling End: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
