from recommend.deployment.Pipeline import Pipeline

item_col = 'sku_cd'
meta_col = 'bom_cd'

pipeline = Pipeline(item_col=item_col, meta_col=meta_col)
pipeline.run()
