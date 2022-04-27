#!/usr/bin/env python
# coding: utf-8

# # Create S3 Bucket

# In[ ]:

#from IPython import get_ipython


import boto3
import sagemaker
import pandas as pd
import numpy as np
import tensorflow

session = boto3.session.Session()
region = session.region_name
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()

s3 = boto3.Session().client(service_name="s3", region_name=region)


# In[ ]:


s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

# In[ ]:


print("Default bucket: {}".format(bucket))


# # Verify S3_BUCKET Bucket Creation

# In[ ]:


#get_ipython().run_cell_magic('bash', '', '\naws s3 ls s3://${bucket}/\n')


# In[ ]:


from botocore.client import ClientError

response = None

try:
    response = s3.head_bucket(Bucket=bucket)
    print(response)
    setup_s3_bucket_passed = True
except ClientError as e:
    print("[ERROR] Cannot find bucket {} in {} due to {}.".format(bucket, response, e))


# In[ ]:


#get_ipython().run_line_magic('store', '')


# # Release Resources

# In[ ]:


#get_ipython().run_cell_magic('html', '', '\n<p><b>Shutting down your kernel for this notebook to release resources.</b></p>\n<button class="sm-command-button" data-commandlinker-command="kernelmenu:shutdown" style="display:none;">Shutdown Kernel</button>\n        \n<script>\ntry {\n    els = document.getElementsByClassName("sm-command-button");\n    els[0].click();\n}\ncatch(err) {\n    // NoOp\n}    \n</script>\n')


# In[ ]:


#git_ipython().run_cell_magic('javascript', '', '\ntry {\n    Jupyter.notebook.save_checkpoint();\n    Jupyter.notebook.session.delete();\n}\ncatch(err) {\n    // NoOp\n}\n')

