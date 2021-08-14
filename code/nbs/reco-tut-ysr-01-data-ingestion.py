#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
project_name = "reco-tut-ysr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

if not os.path.exists(project_path):
    get_ipython().system(u'cp /content/drive/MyDrive/mykeys.py /content')
    import mykeys
    get_ipython().system(u'rm /content/mykeys.py')
    path = "/content/" + project_name; 
    get_ipython().system(u'mkdir "{path}"')
    get_ipython().magic(u'cd "{path}"')
    import sys; sys.path.append(path)
    get_ipython().system(u'git config --global user.email "recotut@recohut.com"')
    get_ipython().system(u'git config --global user.name  "reco-tut"')
    get_ipython().system(u'git init')
    get_ipython().system(u'git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git')
    get_ipython().system(u'git pull origin "{branch}"')
    get_ipython().system(u'git checkout main')
else:
    get_ipython().magic(u'cd "{project_path}"')


# In[5]:


get_ipython().system(u'git status')


# In[6]:


get_ipython().system(u"git add . && git commit -m 'commit' && git push origin main")


# ---

# In[3]:


get_ipython().system(u'cd /content && git clone https://github.com/fafilia/dss_song2vec_recsys.git')


# In[4]:


get_ipython().system(u'mkdir ./data/bronze && cp /content/dss_song2vec_recsys/dataset/yes_complete/* ./data/bronze')

