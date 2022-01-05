import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from recommend.deployment.PipelineCycle import PipelineCycle

cfg = {
    'save_step_yn': True,            # Save each step result to object or csv
    'save_db_yn': True,             #
    'cycle': 'w'
}
# Check start time
print("Start Time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

pipeline = PipelineCycle(cfg=cfg)
pipeline.run()

# Check end time
print("End Time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
