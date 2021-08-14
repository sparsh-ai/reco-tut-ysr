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


# In[24]:


get_ipython().system(u'git status')


# In[25]:


get_ipython().system(u"git add . && git commit -m 'commit' && git push origin main")


# ---

# In[10]:


import pandas as pd
import numpy as np
import itertools
import pickle

import warnings
warnings.filterwarnings("ignore")


# In[13]:


def readTXT(filename, start_line=0, sep=None):
    with open(filename) as file:
        return [line.rstrip().split(sep) for line in file.readlines()[start_line:]]


# ## Song

# In[6]:


songs = pd.read_csv('./data/bronze/song_hash.txt', sep = '\t', header = None,
                    names = ['song_id', 'title', 'artist'], index_col = 0)
songs['artist - title'] = songs['artist'] + " - " + songs['title']
songs.head()


# ## Tag

# In[14]:


tags = readTXT('./data/bronze/tags.txt')
tags[7:12]


# > Note: # means the song doesn't have any tag. we can replace it with unknown

# In[16]:


mapping_tags = dict(readTXT('./data/bronze/tag_hash.txt', sep = ', '))
mapping_tags['#'] = "unknown"
song_tags = pd.DataFrame({'tag_names': [list(map(lambda x: mapping_tags.get(x), t)) for t in tags]})
song_tags.index.name = 'song_id'
song_tags.head()


# We will consider song tags as a feature of song, so will merge it in songs dataset

# In[17]:


songs = pd.merge(left = songs, right = song_tags, how = 'left',
                 left_index = True, right_index = True)
songs.index = songs.index.astype('str')
songs.head()


# We will remove the unknown songs, which doesn't have title and artist. 

# In[18]:


unknown_songs = songs[(songs['artist'] == '-') | (songs['title'] == '-')]
songs.drop(unknown_songs.index, inplace = True)


# ## Playlist

# In[19]:


playlist = readTXT('./data/bronze/train.txt', start_line = 2) + readTXT('./data/bronze/test.txt', start_line = 2)
print(f'Playlist Count: {len(playlist)}')


# In[20]:


for i in range(0, 3):
    print("-------------------------")
    print(f"Playlist Idx. {i}: {len(playlist[i])} Songs")
    print("-------------------------")
    print(playlist[i])


# In[22]:


# Remove unknown songs from the playlist.
playlist_wo_unknown = [[song_id for song_id in p if song_id not in unknown_songs.index]
                       for p in playlist]

# Remove playlist with zero or one song, since the model wouldn't capture any sequence in that list.
clean_playlist = [p for p in playlist_wo_unknown if len(p) > 1]
print(f"Playlist Count After Cleansing: {len(clean_playlist)}")

# Remove song that doesn't exist in any playlist.
unique_songs = set(itertools.chain.from_iterable(clean_playlist))
song_id_not_exist = set(songs.index) - unique_songs
songs.drop(song_id_not_exist, inplace = True)
print(f"Unique Songs After Cleansing: {songs.shape[0]}")


# Before there were 75262 unique songs and 15910 playlists. Now we are ready with 73448 unique songs and 15842 playlists.
# 

# ## Save the artifacts

# In[23]:


get_ipython().system(u'mkdir ./data/silver')

with open('./data/silver/songs.pickle', 'wb') as handle:
    pickle.dump(songs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/silver/clean_playlist.pickle', 'wb') as handle:
    pickle.dump(clean_playlist, handle, protocol=pickle.HIGHEST_PROTOCOL)

