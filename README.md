# Supplementary materials for "Characterizing and Classifying Developer Forum Posts with their Intentions"

## Code
Academic use only.

`model.py`: model definition

`train.py`: code for model training, configurations can be modified.

`predict.py`: generating prediction results for new samples

We also provide a tool to calculate the metrics used in our manuscript.

`python ./tools/evaluation.py -h` shows the usage of the tool.

## About the dataset
Path: `./dataset`
Number of posts: 784
Intention labels are manually annotated.
The source of posts can be identified by the url (`id` key).

## Load data
```
import numpy as np
dataset = np.load(path, allow_pickle=True)
```

## Data format

### Keys
`label`, `id`, `title`, `description`, `description_raw`, `code`, `code_fea`

### Example
```
{'label': ['Errors'],
 'id': 'https://stackoverflow.com/questions/72557738',
 'title': 'Command "python setup.py egg_info" failed with error code 1 in /tmp/pip-build-epojkmk3/duckdb/',
 'description': 'I\'m trying to set up mindsdb in local(visual studio code) with (python version 3.7) using pip3 install mindsdb command but facing an error. Command "python setup.py egg_info" failed with error code 1 in /tmp/pip-build-epojkmk3/duckdb/ How do I resolving this error?',
 'description_raw': "<p>I'm trying to set up mindsdb in local(visual studio code) with (python version 3.7) using</p>\n<pre><code>pip3 install mindsdb \n</code></pre>\n<p>command but facing an error.</p>\n<pre><code>Command &quot;python setup.py egg_info&quot; failed with error code 1 in /tmp/pip-build-epojkmk3/duckdb/\n</code></pre>\n<p>How do I resolving this error?</p>\n",
 'code': ['pip3 install mindsdb \n',
  'Command &quot;python setup.py egg_info&quot; failed with error code 1 in /tmp/pip-build-epojkmk3/duckdb/\n'],
 'code_fea': array([0.01638031, 0.01324874, 0.19329876, 0.13227859, 0.97585787])}
 ```

### Dataset preprocessing

The dataset has been preprocessed. We preprocess the raw HTML using `BeautifulSoup` library. Sample code can be found in `tools` folder.
