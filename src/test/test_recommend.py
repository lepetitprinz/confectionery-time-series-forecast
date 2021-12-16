from recommend.deployment.Pipeline import Pipeline


# Configuration
exec_cfg = {
    'cycle': False,
    'save_step_yn': True
}

pipeline = Pipeline(exec_cfg=exec_cfg)
pipeline.run()
