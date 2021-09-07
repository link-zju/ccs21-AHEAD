# ccs21-AHEAD

## Requirements

numpy==1.19.5

scipy==1.6.2

pandas==1.2.4

treelib==1.6.1

Python>=3.8

## Contents

This project contains 4 folders and 4 example scripts.

1. dataset (folder): The 1-dim or multi-dimensional data records. In each "xxx.txt" file, rows represent user records, and columns represent user attributes. 

2. query_table (folder): the range query table files. Each "xxx.txt" is a random query table which contains 200 random queries. For example, in "Rand_QueryTable_Domain6_Attribute3.txt", the first row is "4 51 20 43 33 40 ". [4, 51] represents the query interval on the first attribute, and [20, 43] represents the query interval on the second attribute.

3. func_module (folder): the basic function modules that AHEAD relies on.

4. rand_result (folder): the query errors when answering the range queries under the MSE metric.

5. ahead_1dim.py (script): the AHEAD algorithm for 1-dim range query.

6. ahead_2dim.py (script): the AHEAD algorithm for 2-dim range query.

7. de_ahead_3dim.py (script): the AHEAD algorithm (Direct Estimation, DE) for 3-dim range query.

9. lle_ahead_highdim.py (script): the AHEAD algorithm (Leveraging Low-dimensional Estimation, LLE) for >2-dim range query.

## Run

```bash
python xxx.py
```

